
import numpy as np
import tensorflow as tf
from tensorflow.python.client.session import Session

from utils.model_utils import (
    keras_serializable,
    get_initializer,
    shape_list,
    TFConv1D,
    TFSharedEmbeddings,
    input_processing
)

from utils.activation import (
    get_activation
)

from utils.output import (
    TFBaseModelOutputWithPast
) 

from config.config_gpt2 import GPT2Config
from utils.model_utils import TFPreTrainedModel

''' 
----- TFAttention -----

@description:

@arg:
    n_ctx(int): Dimensionality of the causal mask (usually same as n_positions)
    n_head(int): Number of heads for multi-head dot-product attention
    split_size(int): 
    scale(bool): whether scale attention weights by dividing by sqrt(hidden_size)
    output_attentions(bool): 
    c_attn(TFCov1D):
    c_proj9TFCov1D):
    attn_dropout(float):
    resid_dropout(float):
    pruned_heads(set):
'''

class TFAttention(tf.keras.layers.Layer):
    def __init__(self, nx, n_ctx, config, scale=False, **kwargs):
        super().__init__(**kwargs)

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implementation]
        assert n_state % config.n_head == 0
        self.n_ctx = n_ctx
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.output_attentions = config.output_attentions

        self.c_attn = TFConv1D(n_state * 3, nx, initializer_range=config.initializer_range, name="c_attn")
        self.c_proj = TFConv1D(n_state, nx, initializer_range=config.initializer_range, name="c_proj")
        self.attn_dropout = tf.keras.layers.Dropout(config.attn_pdrop)
        self.resid_dropout = tf.keras.layers.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        pass

    @staticmethod
    def causal_attention_mask(nd, ns, dtype):
        """
        1's in the lower triangle, counting from the lower right corner. Same as tf.matrix_band_part(tf.ones([nd, ns]),
        -1, ns-nd), but doesn't produce garbage on TPUs.
        """
        i = tf.range(nd)[:, None]
        j = tf.range(ns)
        m = i >= j - ns + nd
        return tf.cast(m, dtype)

    def _attn(self, q, k, v, attention_mask, head_mask, output_attentions, training=False):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        if self.scale:
            dk = tf.cast(shape_list(k)[-1], dtype=w.dtype)  # scale attention_scores
            w = w / tf.math.sqrt(dk)

        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = self.causal_attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w * b - 1e4 * (1 - b)

        if attention_mask is not None:
            # Apply the attention mask
            attention_mask = tf.cast(attention_mask, dtype=w.dtype)
            w = w + attention_mask

        w = tf.nn.softmax(w, axis=-1)
        w = self.attn_dropout(w, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [tf.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = tf.transpose(x, [0, 2, 1, 3])
        x_shape = shape_list(x)
        new_x_shape = x_shape[:-2] + [x_shape[-2] * x_shape[-1]]
        return tf.reshape(x, new_x_shape)

    def split_heads(self, x):
        x_shape = shape_list(x)
        new_x_shape = x_shape[:-1] + [self.n_head, x_shape[-1] // self.n_head]
        x = tf.reshape(x, new_x_shape)
        return tf.transpose(x, (0, 2, 1, 3))  # (batch, head, seq_length, head_features)

    def call(self, x, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=False):
        x = self.c_attn(x)
        query, key, value = tf.split(x, 3, axis=2)
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = tf.unstack(layer_past, axis=0)
            key = tf.concat([past_key, key], axis=-2)
            value = tf.concat([past_value, value], axis=-2)

        # to cope with keras serialization
        if use_cache:
            present = tf.stack([key, value], axis=0)
        else:
            present = (None,)

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions, training=training)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a, training=training)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)

class TFMLP(tf.keras.layers.Layer):
    def __init__(self, n_state, config, **kwargs):
        super().__init__(**kwargs)
        nx = config.n_embd
        self.c_fc = TFConv1D(n_state, nx, initializer_range=config.initializer_range, name="c_fc")
        self.c_proj = TFConv1D(nx, n_state, initializer_range=config.initializer_range, name="c_proj")
        self.act = get_activation("gelu")
        self.dropout = tf.keras.layers.Dropout(config.resid_pdrop)

    def call(self, x, training=False):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        h2 = self.dropout(h2, training=training)
        return h2


class TFBlock(tf.keras.layers.Layer):
    def __init__(self, n_ctx, config, scale=False, **kwargs):
        super().__init__(**kwargs)
        nx = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * nx
        self.ln_1 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_1")
        self.attn = TFAttention(nx, n_ctx, config, scale, name="attn")
        self.ln_2 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_2")
        self.mlp = TFMLP(inner_dim, config, name="mlp")

    def call(self, x, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=False):
        a = self.ln_1(x)
        output_attn = self.attn(
            a, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=training
        )
        a = output_attn[0]  # output_attn: a, present, (attentions)
        x = x + a

        m = self.ln_2(x)
        m = self.mlp(m, training=training)
        x = x + m

        outputs = [x] + output_attn[1:]
        return outputs  # x, present, (attentions)

class TFImageTransformer(tf.keras.layers.Layer):
    def __init__(self, nx, initializer_range, last_dim: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.nx = nx
        self.initializer_range = initializer_range
        self.last_dim = last_dim

    def build(self, input_shape = None):
        self.transformer = self.add_weight(
            "transformer", shape = [self.nx, self.last_dim], initializer=get_initializer(self.initializer_range)
        )

    def call(self, inputs):
        bz, sl = shape_list(inputs)[:2]

        x = tf.reshape(inputs, [-1, self.nx])
        x = tf.matmul(x, self.transformer)
        last_dim = shape_list(self.transformer)[-1]

        size = int(sl ** 0.5)

        return tf.reshape(x, [bz, size, size, last_dim])



@keras_serializable
class TFGPT2MainLayer(tf.keras.layers.Layer):
    config_class = GPT2Config

    def __init__(self, config, last_dim: int = 1, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.use_cache = config.use_cache
        self.return_dict = config.use_return_dict

        self.num_hidden_layers = config.n_layer
        self.vocab_size = config.vocab_size
        self.n_embd = config.n_embd
        self.n_positions = config.n_positions
        self.initializer_range = config.initializer_range

        self.wte = TFSharedEmbeddings(
            config.vocab_size, config.hidden_size, initializer_range=config.initializer_range, name="wte"
        )
        self.drop = tf.keras.layers.Dropout(config.embd_pdrop)
        self.h = [TFBlock(config.n_ctx, config, scale=True, name=f"h_._{i}") for i in range(config.n_layer)]
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_f")
        self.activation_layer = tf.keras.layers.ReLU(max_value = 1)
        self.transformer = TFImageTransformer(config.n_embd, config.initializer_range, last_dim)

    def build(self, input_shape):
        with tf.name_scope("wpe"):
            self.wpe = self.add_weight(
                name="embeddings",
                shape=[self.n_positions, self.n_embd],
                initializer=get_initializer(self.initializer_range),
            )

        super().build(input_shape)

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value):
        self.wte.weight = value
        self.wte.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        raise NotImplementedError
        
    def call(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs,
    ):

        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        if inputs["input_ids"] is not None and inputs["inputs_embeds"] is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif inputs["input_ids"] is not None:
            input_shape = shape_list(inputs["input_ids"])
            inputs["input_ids"] = input_ids
            # inputs["input_ids"] = tf.reshape(inputs["input_ids"], [-1, input_shape[-1]])
        elif inputs["inputs_embeds"] is not None:
            input_shape = shape_list(inputs["inputs_embeds"])[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs["past"] is None:
            past_length = 0
            inputs["past"] = [None] * len(self.h)
        else:
            past_length = shape_list(inputs["past"][0][0])[-2]

        if inputs["position_ids"] is None:
            inputs["position_ids"] = tf.expand_dims(tf.range(past_length, input_shape[-2] + past_length), axis=0)

        if inputs["attention_mask"] is not None:
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask_shape = shape_list(inputs["attention_mask"])
            inputs["attention_mask"] = tf.reshape(
                inputs["attention_mask"], (attention_mask_shape[0], 1, 1, attention_mask_shape[1])
            )

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            one_cst = tf.constant(1.0)
            inputs["attention_mask"] = tf.cast(inputs["attention_mask"], dtype=one_cst.dtype)
            inputs["attention_mask"] = tf.multiply(
                tf.subtract(one_cst, inputs["attention_mask"]), tf.constant(-10000.0)
            )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if inputs["head_mask"] is not None:
            raise NotImplementedError
        else:
            inputs["head_mask"] = [None] * self.num_hidden_layers
            # head_mask = tf.constant([0] * self.num_hidden_layers)

        inputs["position_ids"] = tf.reshape(inputs["position_ids"], [shape_list(inputs["position_ids"])[-1]])

        if inputs["inputs_embeds"] is None:
            inputs["inputs_embeds"] = inputs["input_ids"]
            # inputs["inputs_embeds"] = self.wte(inputs["input_ids"], mode="embedding")

        # print(inputs["position_ids"])
        position_embeds = tf.gather(self.wpe, inputs["position_ids"])

        if inputs["token_type_ids"] is not None:
            inputs["token_type_ids"] = tf.reshape(
                inputs["token_type_ids"], [-1, shape_list(inputs["token_type_ids"])[-1]]
            )
            token_type_embeds = self.wte(inputs["token_type_ids"], mode="embedding")
        else:
            token_type_embeds = tf.constant(0.0)

        # print("inputs_embeds")
        # print(inputs["inputs_embeds"])
        # print(position_embeds)

        position_embeds = tf.cast(position_embeds, dtype=inputs["inputs_embeds"].dtype)
        token_type_embeds = tf.cast(token_type_embeds, dtype=inputs["inputs_embeds"].dtype)
        # print("-" * 80)
        # print(position_embeds)
        # print(token_type_embeds)
        # print(inputs["inputs_embeds"])
        hidden_states = inputs["inputs_embeds"] + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states, training=inputs["training"])

        # output_shape = input_shape + [shape_list(hidden_states)[-1]]
        # print("input_shape: {}".format(input_shape))
        output_shape = input_shape

        presents = () if inputs["use_cache"] else None
        all_attentions = () if inputs["output_attentions"] else None
        all_hidden_states = () if inputs["output_hidden_states"] else None

        for i, (block, layer_past) in enumerate(zip(self.h, inputs["past"])):
            if inputs["output_hidden_states"]:
                all_hidden_states = all_hidden_states + (tf.reshape(hidden_states, output_shape),)

            outputs = block(
                hidden_states,
                layer_past,
                inputs["attention_mask"],
                inputs["head_mask"][i],
                inputs["use_cache"],
                inputs["output_attentions"],
                training=inputs["training"],
            )

            hidden_states, present = outputs[:2]
            if inputs["use_cache"]:
                presents = presents + (present,)

            if inputs["output_attentions"]:
                all_attentions = all_attentions + (outputs[2],)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = tf.reshape(hidden_states, output_shape)
        # print("-" * 100)
        # print("output_shape: {}".format(output_shape))
        
        # Add last hidden state
        if inputs["output_hidden_states"]:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if inputs["output_attentions"]:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + [-1] + shape_list(all_attentions[0])[-2:]
            all_attentions = tuple(tf.reshape(t, attention_output_shape) for t in all_attentions)

        if not inputs["return_dict"]:
            print("return dict")
            print(tuple(v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None))
            print('-' * 80)
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None)

        hidden_states = self.activation_layer(hidden_states)
        hidden_states = self.transformer(hidden_states)

        return hidden_states

class TFGPT2PreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPT2Config
    base_model_prefix = "transformer"
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"h.\d+.attn.bias"]

    @tf.function(
        input_signature=[
            {
                "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),
                "attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),
            }
        ]
    )
    def serving(self, inputs):
        output = self.call(inputs)

        return self.serving_output(output)