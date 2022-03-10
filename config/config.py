
import os
import json

from logging import error
from .config_utils import PretrainedConfig

'''
----- TrainingConfig -----
@desciption:
    This is the configuration class to store the configuration of a :class:`~transformers.GPT2Model` or a
    :class:`~transformers.TFGPT2Model`. It is used to instantiate a GPT-2 model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the GPT-2 `small <https://huggingface.co/gpt2>`__ architecture.

@args:

    ...... NOT COMPLETE ......

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

class TrainingConfig(PretrainedConfig):

    def __init__(self):

        self.load_config("./config/config.json")

        # with open("./config/config.json") as f:
        #     d = json.load(f)

        # self.gpu = bool(d["gpu"])
        # self.parallel = bool(d["parallel"])

        # self.subject_count = int(d["subject_count"])

        # self.model_name = str(d["model_name"])
        # self.buffer_size = int(d["buffer_size"])
        # self.batch_size = int(d["batch_size"])
        # self.epoches = int(d["epoches"])
        # self.learning_rate = float(d["learning_rate"])
        # self.example_to_generate = int(d["example_to_generate"])
        # self.noise_hidden_dim = int(d["noise_hidden_dim"])

        # self.vocab_size = int(d["vocab_size"])
        # self.n_positions = int(d["n_positions"])
        # self.n_ctx = int(d["n_ctx"])
        # self.n_embd = int(d["n_embd"])
        # self.n_layer = int(d["n_layer"])
        # self.n_head = int(d["n_head"])
        # self.n_inner = None if str(d["n_inner"]) == "None" else int(d["n_inner"])
        # self.activation_function = str(d["activation_function"])
        # self.resid_pdrop = float(d["resid_pdrop"])
        # self.embd_pdrop = float(d["embd_pdrop"])
        # self.attn_pdrop = float(d["attn_pdrop"])
        # self.layer_norm_epsilon = float(d["layer_norm_epsilon"])
        # self.initializer_range = float(d["initializer_range"])
        # self.summary_type = str(d["summary_type"])
        # self.summary_use_proj = bool(d["summary_use_proj"]),
        # self.summary_activation = None if str(d["summary_activation"]) == "None" else str(d["summary_activation"])
        # self.summary_proj_to_labels = bool(d["summary_proj_to_labels"])
        # self.summary_first_dropout = float(d["summary_first_dropout"])
        # self.scale_attn_weights = bool(d["scale_attn_weights"])
        # self.gradient_checkpointing = bool(d["gradient_checkpointing"])
        # self.use_cache = bool(d["use_cache"])
        # self.bos_token_id = int(d["bos_token_id"])
        # self.eos_token_id = int(d["eos_token_id"])

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

    def save_config(self, model_name, time):
        
        file_path = "./result/{}/{}/config/config.txt".format(model_name, time)

        data = {
            "gpu": self.gpu, 
            "gpu_id": self.gpu_id,
            "parallel": self.parallel,

            "subject_count": self.subject_count,
            "data_path": self.data_path,

            "model_name": self.model_name,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "epoches": self.epoches,
            "rounds": self.rounds,
            "learning_rate": self.learning_rate,
            "random_vector_num": self.random_vector_num,
            "example_to_generate": self.example_to_generate,
            "condition_size": self.condition_size,
            "class_count": self.class_count,
            "noise_variance": self.noise_variance,
            "fine_tune": self.fine_tune,
            "pretrained_finetune_path": self.pretrained_finetune_path,
            "pretrained_classifier_path": self.pretrained_classifier_path,

            "vocab_size": self.vocab_size,
            "n_positions": self.n_positions,
            "n_ctx": self.n_ctx,
            "n_embd": self.n_embd,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_inner": self.n_inner,
            "activation_function": self.activation_function,
            "resid_pdrop": self.resid_pdrop,
            "embd_pdrop": self.embd_pdrop,
            "attn_pdrop": self.attn_pdrop,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "initializer_range": self.initializer_range,
            "summary_type": self.summary_type,
            "summary_use_proj": self.summary_use_proj,
            "summary_activation": self.summary_activation,
            "summary_proj_to_labels": self.summary_proj_to_labels,
            "summary_first_dropout": self.summary_first_dropout,
            "scale_attn_weights": self.scale_attn_weights,
            "gradient_checkpointing": self.gradient_checkpointing,
            "use_cache": self.use_cache,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }

        jsonStr = json.dumps(data)

        with open(file = file_path, mode = 'w') as f:
            f.write(jsonStr)

    def load_config(self, path = None):
        
        if path is None:
            error("[TrainingConfig] load_config needs fild path for loading!!!")
            return None
        else:
            if not os.path.exists(path):
                error("[TrainingConfig] load_config needs VALID file path for laoding!!!")
                return None
            else:
                with open(path) as f:
                    d = json.load(f)

                self.gpu = bool(d["gpu"])
                self.gpu_id = int(d["gpu_id"])
                self.parallel = bool(d["parallel"])

                self.subject_count = int(d["subject_count"])
                self.data_path = str(d["data_path"])

                self.model_name = str(d["model_name"])
                self.buffer_size = int(d["buffer_size"])
                self.batch_size = int(d["batch_size"])
                self.epoches = int(d["epoches"])
                self.rounds = int(d["rounds"])
                self.learning_rate = float(d["learning_rate"])
                self.random_vector_num = int(d["random_vector_num"])
                self.example_to_generate = int(d["example_to_generate"])
                self.condition_size = int(d["condition_size"])
                self.class_count = int(d["class_count"])
                self.noise_variance = float(d["noise_variance"])
                self.fine_tune = bool(d["fine_tune"])
                self.pretrained_finetune_path = str(d["pretrained_finetune_path"])
                self.pretrained_classifier_path = str(d["pretrained_classifier_path"])

                self.vocab_size = int(d["vocab_size"])
                self.n_positions = int(d["n_positions"])
                self.n_ctx = int(d["n_ctx"])
                self.n_embd = int(d["n_embd"])
                self.n_layer = int(d["n_layer"])
                self.n_head = int(d["n_head"])
                self.n_inner = None if str(d["n_inner"]) == "None" else int(d["n_inner"])
                self.activation_function = str(d["activation_function"])
                self.resid_pdrop = float(d["resid_pdrop"])
                self.embd_pdrop = float(d["embd_pdrop"])
                self.attn_pdrop = float(d["attn_pdrop"])
                self.layer_norm_epsilon = float(d["layer_norm_epsilon"])
                self.initializer_range = float(d["initializer_range"])
                self.summary_type = str(d["summary_type"])
                self.summary_use_proj = bool(d["summary_use_proj"])
                self.summary_activation = None if str(d["summary_activation"]) == "None" else str(d["summary_activation"])
                self.summary_proj_to_labels = bool(d["summary_proj_to_labels"])
                self.summary_first_dropout = float(d["summary_first_dropout"])
                self.scale_attn_weights = bool(d["scale_attn_weights"])
                self.gradient_checkpointing = bool(d["gradient_checkpointing"])
                self.use_cache = bool(d["use_cache"])
                self.bos_token_id = int(d["bos_token_id"])
                self.eos_token_id = int(d["eos_token_id"])