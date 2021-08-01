
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from logging import error
from .config_utils import PretrainedConfig

'''
----- GPT2Config -----
@desciption:
    This is the configuration class to store the configuration of a :class:`~transformers.GPT2Model` or a
    :class:`~transformers.TFGPT2Model`. It is used to instantiate a GPT-2 model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the GPT-2 `small <https://huggingface.co/gpt2>`__ architecture.

@args:
    vocab_size (:obj:`int`, `optional`, defaults to 50257):
        Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
        :obj:`inputs_ids` passed when calling :class:`~transformers.GPT2Model` or
        :class:`~transformers.TFGPT2Model`.
    n_positions (:obj:`int`, `optional`, defaults to 1024):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    n_ctx (:obj:`int`, `optional`, defaults to 1024):
        Dimensionality of the causal mask (usually same as n_positions).
    n_embd (:obj:`int`, `optional`, defaults to 768):
        Dimensionality of the embeddings and hidden states.
    n_layer (:obj:`int`, `optional`, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    n_head (:obj:`int`, `optional`, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    n_inner (:obj:`int`, `optional`, defaults to None):
        Dimensionality of the inner feed-forward layers. :obj:`None` will set it to 4 times n_embd
    activation_function (:obj:`str`, `optional`, defaults to :obj:`"gelu"`):
        Activation function, to be selected in the list :obj:`["relu", "silu", "gelu", "tanh", "gelu_new"]`.
    resid_pdrop (:obj:`float`, `optional`, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    embd_pdrop (:obj:`int`, `optional`, defaults to 0.1):
        The dropout ratio for the embeddings.
    attn_pdrop (:obj:`float`, `optional`, defaults to 0.1):
        The dropout ratio for the attention.
    layer_norm_epsilon (:obj:`float`, `optional`, defaults to 1e-5):
        The epsilon to use in the layer normalization layers
    initializer_range (:obj:`float`, `optional`, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    summary_type (:obj:`string`, `optional`, defaults to :obj:`"cls_index"`):
        Argument used when doing sequence summary, used in the models :class:`~transformers.GPT2DoubleHeadsModel`
        and :class:`~transformers.TFGPT2DoubleHeadsModel`.

        Has to be one of the following options:

            - :obj:`"last"`: Take the last token hidden state (like XLNet).
            - :obj:`"first"`: Take the first token hidden state (like BERT).
            - :obj:`"mean"`: Take the mean of all tokens hidden states.
            - :obj:`"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
            - :obj:`"attn"`: Not implemented now, use multi-head attention.
    summary_use_proj (:obj:`bool`, `optional`, defaults to :obj:`True`):
        Argument used when doing sequence summary, used in the models :class:`~transformers.GPT2DoubleHeadsModel`
        and :class:`~transformers.TFGPT2DoubleHeadsModel`.

        Whether or not to add a projection after the vector extraction.
    summary_activation (:obj:`str`, `optional`):
        Argument used when doing sequence summary. Used in for the multiple choice head in
        :class:`~transformers.GPT2DoubleHeadsModel`.

        Pass :obj:`"tanh"` for a tanh activation to the output, any other value will result in no activation.
    summary_proj_to_labels (:obj:`bool`, `optional`, defaults to :obj:`True`):
        Argument used when doing sequence summary, used in the models :class:`~transformers.GPT2DoubleHeadsModel`
        and :class:`~transformers.TFGPT2DoubleHeadsModel`.

        Whether the projection outputs should have :obj:`config.num_labels` or :obj:`config.hidden_size` classes.
    summary_first_dropout (:obj:`float`, `optional`, defaults to 0.1):
        Argument used when doing sequence summary, used in the models :class:`~transformers.GPT2DoubleHeadsModel`
        and :class:`~transformers.TFGPT2DoubleHeadsModel`.

        The dropout ratio to be used after the projection and activation.
    scale_attn_weights (:obj:`bool`, `optional`, defaults to :obj:`True`):
        Scale attention weights by dividing by sqrt(hidden_size).
    gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.
    use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
        Whether or not the model should return the last key/values attentions (not used by all models).
'''

class GPT2Config(PretrainedConfig):
    
    model_type = "gpt2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=4,
        n_head=12,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        gradient_checkpointing=False,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        **kwargs
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.gradient_checkpointing = gradient_checkpointing
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    @property
    def max_position_embeddings(self):
        return self.n_positions

    @property
    def hidden_size(self):
        return self.n_embd

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer

    
def save_model_config(config, model_name, time, current_round):

    file_path = "./trained_model/" + str(model_name) + "/" + str(time) + "/model_" + str(current_round) + ".txt"

    data = {
        "vocab_size": config.vocab_size,
        "n_positions": config.n_postions,
        "n_ctx": config.n_ctx,
        "n_embd": config.n_embd,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_inner": config.n_inner,
        "activation_function": config.activation_function,
        "resid_pdrop": config.resid_pdrop,
        "embd_pdrop": config.embd_pdrop,
        "attn_pdrop": config.attn_pdrop,
        "layer_norm_epsilon": config.layer_norm_epsilon,
        "initializer_range": config.initializer_range,
        "summary_type": config.summary_type,
        "summary_use_proj": config.summary_use_proj,
        "summary_activation": config.summary_activation,
        "summary_proj_to_labels": config.summary_proj_to_labels,
        "summary_first_dropout": config.summary_first_dropout,
        "scale_attn_weights": config.scale_attn_weights,
        "gradient_checkpointing": config.gradient_checkpointing,
        "use_cache": config.use_cache,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
    }

    jsonStr = json.dumps(data)

    with open(file = file_path, mode = 'w') as f:
        f.write(jsonStr)

def load_model_config(model_path):

    if not os.path.exists(model_path):
        error("[model_monitor.py] load_model_config: given model path is invalid")
        return None
    else:
        with open(model_path) as f:
            d = json.load(f)

        config = GPT2Config(
            vocab_size = int(d["vocab_size"]),
            n_positions = int(d["n_positions"]),
            n_ctx = int(d["n_ctx"]),
            n_embd = int(d["n_embd"]),
            n_layer = int(d["n_layer"]),
            n_head = int(d["n_head"]),
            n_inner = None if str(d["n_inner"]) == "None" else int(d["n_inner"]),
            activation_function = str(d["activation_function"]),
            resid_pdrop = float(d["resid_pdrop"]),
            embd_pdrop = float(d["embd_pdrop"]),
            attn_pdrop = float(d["attn_pdrop"]),
            layer_norm_epsilon = float(d["layer_norm_epsilon"]),
            initializer_range = float(d["initializer_range"]),
            summary_type = str(d["summary_type"]),
            summary_use_proj = bool(d["summary_use_proj"]),
            summary_activation = None if str(d["summary_activation"]) == "None" else str(d["summary_activation"]),
            summary_proj_to_labels = bool(d["summary_proj_to_labels"]),
            summary_first_dropout = float(d["summary_first_dropout"]),
            scale_attn_weights = bool(d["scale_attn_weights"]),
            gradient_checkpointing = bool(d["gradient_checkpointing"]),
            use_cache = bool(d["use_cache"]),
            bos_token_id = int(d["bos_token_id"]),
            eos_token_id = int(d["eos_token_id"]),
        )

        return config


