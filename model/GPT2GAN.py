
import os
import tensorflow as tf

from .gpt2 import TFGPT2MainLayer
from .discriminator import Discriminator
from utils.output import TFCausalLMOutputWithPast
from utils.model_utils import input_processing, TFPreTrainedModel, TFCausalLanguageModelingLoss

class GPT2GAN(TFPreTrainedModel, TFCausalLanguageModelingLoss):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.generator = TFGPT2MainLayer(config, name = "generator")
        self.discriminator = Discriminator(config, name = "discriminator")

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    def get_output_embeddings(self):
        return self.get_input_embeddings()

    def set_output_embeddings(self, value):
        self.set_input_embeddings(value)

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
        labels=None,
        training=False,
        **kwargs,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the cross entropy classification loss. Indices should be in ``[0, ...,
            config.vocab_size - 1]``.
        """
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
            labels=labels,
            training=training,
            kwargs_call=kwargs,
        )

        generator_outputs = self.generator(
            input_ids=inputs["input_ids"],
            past=inputs["past"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            use_cache=inputs["use_cache"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )

        hidden_states = generator_outputs[0]
        logits = self.generator.wte(hidden_states, mode="linear")

        loss = None

        if inputs["labels"] is not None:
            # shift labels to the left and cut last logit token
            logits = logits[:, :-1]
            labels = inputs["labels"][:, 1:]
            loss = self.compute_loss(labels, logits)

        if not inputs["return_dict"]:
            output = (logits,) + generator_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFCausalLMOutputWithPast(
            loss = loss,
            logits = logits,
            past_key_values = generator_outputs.past_key_values,
            hidden_states = generator_outputs.hidden_states,
            attentions = generator_outputs.attentions,
        )