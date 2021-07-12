
import numpy as np
import tensorflow as tf

from typing import List, Optional, Union, Dict

import functools
import warnings
import inspect

from config_utils import PretrainedConfig
from tokenization_utils_base import BatchEncoding

from file_utils import (
    ModelOutput
)

import logging
logger = logging.get_logger(__name__)
tf_logger = tf.get_logger()

TFModelInputType = Union[
    List[tf.Tensor], List[np.ndarray], Dict[str, tf.Tensor], Dict[str, np.ndarray], np.ndarray, tf.Tensor
]

def keras_serializable(cls):
    """
    Decorate a Keras Layer class to support Keras serialization.

    This is done by:

    1. Adding a :obj:`transformers_config` dict to the Keras config dictionary in :obj:`get_config` (called by Keras at
       serialization time.
    2. Wrapping :obj:`__init__` to accept that :obj:`transformers_config` dict (passed by Keras at deserialization
       time) and convert it to a config object for the actual layer initializer.
    3. Registering the class as a custom object in Keras (if the Tensorflow version supports this), so that it does not
       need to be supplied in :obj:`custom_objects` in the call to :obj:`tf.keras.models.load_model`.

    Args:
        cls (a :obj:`tf.keras.layers.Layers subclass`):
            Typically a :obj:`TF.MainLayer` class in this project, in general must accept a :obj:`config` argument to
            its initializer.

    Returns:
        The same class object, with modifications for Keras deserialization.
    """
    initializer = cls.__init__

    config_class = getattr(cls, "config_class", None)
    if config_class is None:
        raise AttributeError("Must set `config_class` to use @keras_serializable")

    @functools.wraps(initializer)
    def wrapped_init(self, *args, **kwargs):
        config = args[0] if args and isinstance(args[0], PretrainedConfig) else kwargs.pop("config", None)

        if isinstance(config, dict):
            config = config_class.from_dict(config)
            initializer(self, config, *args, **kwargs)
        elif isinstance(config, PretrainedConfig):
            if len(args) > 0:
                initializer(self, *args, **kwargs)
            else:
                initializer(self, config, *args, **kwargs)
        else:
            raise ValueError("Must pass either `config` (PretrainedConfig) or `config` (dict)")

        self._config = config
        self._kwargs = kwargs

    cls.__init__ = wrapped_init

    if not hasattr(cls, "get_config"):
        raise TypeError("Only use @keras_serializable on tf.keras.layers.Layer subclasses")
    if hasattr(cls.get_config, "_is_default"):

        def get_config(self):
            cfg = super(cls, self).get_config()
            cfg["config"] = self._config.to_dict()
            cfg.update(self._kwargs)
            return cfg

        cls.get_config = get_config

    cls._keras_serializable = True
    if hasattr(tf.keras.utils, "register_keras_serializable"):
        cls = tf.keras.utils.register_keras_serializable()(cls)
    return cls


def get_initializer(initializer_range: float = 0.02) -> tf.initializers.TruncatedNormal:
    """
    Creates a :obj:`tf.initializers.TruncatedNormal` with the given range.

    Args:
        initializer_range (`float`, defaults to 0.02): Standard deviation of the initializer range.

    Returns:
        :obj:`tf.initializers.TruncatedNormal`: The truncated normal initializer.
    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)

def shape_list(tensor: tf.Tensor) -> List[int]:
    """
    Deal with dynamic shape in tensorflow cleanly.

    Args:
        tensor (:obj:`tf.Tensor`): The tensor we want the shape of.

    Returns:
        :obj:`List[int]`: The shape of the tensor as a list.
    """
    dynamic = tf.shape(tensor)

    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()

    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

class TFConv1D(tf.keras.layers.Layer):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (:obj:`int`):
            The number of output features.
        nx (:obj:`int`):
            The number of input features.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation to use to initialize the weights.
        kwargs:
            Additional keyword arguments passed along to the :obj:`__init__` of :obj:`tf.keras.layers.Layer`.
    """

    def __init__(self, nf, nx, initializer_range=0.02, **kwargs):
        super().__init__(**kwargs)
        self.nf = nf
        self.nx = nx
        self.initializer_range = initializer_range

    def build(self, input_shape):
        self.weight = self.add_weight(
            "weight", shape=[self.nx, self.nf], initializer=get_initializer(self.initializer_range)
        )
        self.bias = self.add_weight("bias", shape=[1, self.nf], initializer=tf.zeros_initializer())

    def call(self, x):
        bz, sl = shape_list(x)[:2]

        x = tf.reshape(x, [-1, self.nx])
        x = tf.matmul(x, self.weight) + self.bias

        x = tf.reshape(x, [bz, sl, self.nf])

        return x


class TFSharedEmbeddings(tf.keras.layers.Layer):
    r"""
    Construct shared token embeddings.

    The weights of the embedding layer is usually shared with the weights of the linear decoder when doing language
    modeling.

    Args:
        vocab_size (:obj:`int`):
            The size of the vocabulary, e.g., the number of unique tokens.
        hidden_size (:obj:`int`):
            The size of the embedding vectors.
        initializer_range (:obj:`float`, `optional`):
            The standard deviation to use when initializing the weights. If no value is provided, it will default to
            :math:`1/\sqrt{hidden\_size}`.
        kwargs:
            Additional keyword arguments passed along to the :obj:`__init__` of :obj:`tf.keras.layers.Layer`.
    """

    def __init__(self, vocab_size: int, hidden_size: int, initializer_range: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.initializer_range = hidden_size ** -0.5 if initializer_range is None else initializer_range

    def build(self, input_shape):
        """
        Build shared token embedding layer Shared weights logic adapted from
        https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        self.weight = self.add_weight(
            "weight", shape=[self.vocab_size, self.hidden_size], initializer=get_initializer(self.initializer_range)
        )
        super().build(input_shape)

    def get_config(self):
        config = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "initializer_range": self.initializer_range,
        }
        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs: tf.Tensor, mode: str = "embedding") -> tf.Tensor:
        """
        Get token embeddings of inputs or decode final hidden state.

        Args:
            inputs (:obj:`tf.Tensor`):
                In embedding mode, should be an int64 tensor with shape :obj:`[batch_size, length]`.

                In linear mode, should be a float tensor with shape :obj:`[batch_size, length, hidden_size]`.
            mode (:obj:`str`, defaults to :obj:`"embedding"`):
               A valid value is either :obj:`"embedding"` or :obj:`"linear"`, the first one indicates that the layer
               should be used as an embedding layer, the second one that the layer should be used as a linear decoder.

        Returns:
            :obj:`tf.Tensor`: In embedding mode, the output is a float32 embedding tensor, with shape
            :obj:`[batch_size, length, embedding_size]`.

            In linear mode, the output is a float32 with shape :obj:`[batch_size, length, vocab_size]`.

        Raises:
            ValueError: if :obj:`mode` is not valid.

        Shared weights logic is adapted from `here
        <https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24>`__.
        """
        if mode == "embedding":
            return self._embedding(inputs)
        elif mode == "linear":
            return self._linear(inputs)
        else:
            raise ValueError(f"mode {mode} is not valid.")

    def _embedding(self, input_ids):
        """Applies embedding based on inputs tensor."""
        return tf.gather(self.weight, input_ids)

    def _linear(self, inputs):
        """
        Computes logits by running inputs through a linear layer.

        Args:
            inputs: A float32 tensor with shape [..., hidden_size]

        Returns:
            float32 tensor with shape [..., vocab_size].
        """
        first_dims = shape_list(inputs)[:-1]
        x = tf.reshape(inputs, [-1, self.hidden_size])
        logits = tf.matmul(x, self.weight, transpose_b=True)

        return tf.reshape(logits, first_dims + [self.vocab_size])

def booleans_processing(config, **kwargs):
    """
    Process the input booleans of each model in order to be sure they are compliant with the execution mode (eager or
    graph)

    Args:
        config (:class:`~transformers.PretrainedConfig`):
            The config of the running model.
        **kwargs:
            The boolean parameters

    Returns:
        A dictionary with the proper values for each boolean
    """
    final_booleans = {}

    if tf.executing_eagerly():
        final_booleans["output_attentions"] = (
            kwargs["output_attentions"] if kwargs["output_attentions"] is not None else config.output_attentions
        )
        final_booleans["output_hidden_states"] = (
            kwargs["output_hidden_states"]
            if kwargs["output_hidden_states"] is not None
            else config.output_hidden_states
        )
        final_booleans["return_dict"] = (
            kwargs["return_dict"] if kwargs["return_dict"] is not None else config.return_dict
        )

        if "use_cache" in kwargs:
            final_booleans["use_cache"] = kwargs["use_cache"] if kwargs["use_cache"] is not None else config.use_cache
    else:
        if (
            kwargs["output_attentions"] is not None
            or kwargs["output_hidden_states"] is not None
            or ("use_cache" in kwargs and kwargs["use_cache"] is not None)
        ):
            tf_logger.warning(
                "The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model."
                "They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`)."
            )

        final_booleans["output_attentions"] = config.output_attentions
        final_booleans["output_hidden_states"] = config.output_hidden_states

        if kwargs["return_dict"] is not None:
            tf_logger.warning(
                "The parameter `return_dict` cannot be set in graph mode and will always be set to `True`."
            )
        final_booleans["return_dict"] = True

        if "use_cache" in kwargs:
            final_booleans["use_cache"] = config.use_cache

    return final_booleans

def input_processing(func, config, input_ids, **kwargs):
    """
    Process the input of each TensorFlow model including the booleans. In case of a list of symbolic inputs, each input
    has to be named accordingly to the parameters name, i.e. `input_ids = tf.keras.Input(shape=(128,), dtype='int32',
    name="input_ids")` otherwise the order of the tensors will not be guaranteed during the training.

    Args:
        func (:obj:`callable`):
            The callable function of the TensorFlow model.
        config (:class:`~transformers.PretrainedConfig`):
            The config of the running model.
        **kwargs:
            The inputs of the model.

    Returns:
        Two lists, one for the missing layers, and another one for the unexpected layers.
    """
    signature = dict(inspect.signature(func).parameters)
    signature.pop("kwargs", None)
    signature.pop("self", None)
    parameter_names = list(signature.keys())
    output = {}
    allowed_types = (tf.Tensor, bool, int, ModelOutput, tuple, list, dict, np.ndarray)

    if "inputs" in kwargs["kwargs_call"]:
        warnings.warn(
            "The `inputs` argument is deprecated and will be removed in a future version, use `input_ids` instead.",
            FutureWarning,
        )

        output["input_ids"] = kwargs["kwargs_call"].pop("inputs")

    if "decoder_cached_states" in kwargs["kwargs_call"]:
        warnings.warn(
            "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
            FutureWarning,
        )
        output["past_key_values"] = kwargs["kwargs_call"].pop("decoder_cached_states")

    if len(kwargs["kwargs_call"]) > 0:
        raise ValueError(
            f"The following keyword arguments are not supported by this model: {list(kwargs['kwargs_call'].keys())}."
        )

    kwargs.pop("kwargs_call")

    for k, v in kwargs.items():
        if isinstance(v, allowed_types) or v is None:
            output[k] = v
        else:
            raise ValueError(f"Data of type {type(v)} is not allowed only {allowed_types} is accepted for {k}.")

    if isinstance(input_ids, (tuple, list)):
        for i, input in enumerate(input_ids):
            # EagerTensors don't allow to use the .name property so we check for a real Tensor
            if type(input) == tf.Tensor:
                # Tensor names have always the pattern `name:id` then we check only the
                # `name` part
                tensor_name = input.name.split(":")[0]

                if tensor_name in parameter_names:
                    output[tensor_name] = input
                else:
                    output[parameter_names[i]] = input
            elif isinstance(input, allowed_types) or input is None:
                output[parameter_names[i]] = input
            else:
                raise ValueError(
                    f"Data of type {type(input)} is not allowed only {allowed_types} is accepted for {parameter_names[i]}."
                )
    elif isinstance(input_ids, (dict, BatchEncoding)):
        if "inputs" in input_ids:
            warnings.warn(
                "The `inputs` argument is deprecated and will be removed in a future version, use `input_ids` instead.",
                FutureWarning,
            )

            output["input_ids"] = input_ids.pop("inputs")

        if "decoder_cached_states" in input_ids:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            output["past_key_values"] = input_ids.pop("decoder_cached_states")

        for k, v in dict(input_ids).items():
            if isinstance(v, allowed_types) or v is None:
                output[k] = v
            elif k not in parameter_names and "args" not in parameter_names:
                logger.warning(
                    f"The parameter {k} does not belongs to the parameter list {parameter_names} and will be ignored."
                )
                continue
            else:
                raise ValueError(f"Data of type {type(v)} is not allowed only {allowed_types} is accepted for {k}.")
    else:
        if isinstance(input_ids, tf.Tensor) or input_ids is None:
            output[parameter_names[0]] = input_ids
        else:
            raise ValueError(
                f"Data of type {type(input_ids)} is not allowed only {allowed_types} is accepted for {parameter_names[0]}."
            )

    for name in parameter_names:
        if name not in list(output.keys()) and name != "args":
            output[name] = kwargs.pop(name, signature[name].default)

    # When creating a SavedModel TF calls the method with LayerCall.__call__(args, **kwargs)
    # So to respect the proper output we have to add this exception
    if "args" in output:
        if output["args"] is not None and type(output["args"]) == tf.Tensor:
            tensor_name = output["args"].name.split(":")[0]
            output[tensor_name] = output["args"]
        else:
            # `args` in this case is always the first parameter, then `input_ids`
            output["input_ids"] = output["args"]

        del output["args"]

    if "kwargs" in output:
        del output["kwargs"]

    boolean_dict = {
        k: v
        for k, v in output.items()
        if k in ["return_dict", "output_attentions", "output_hidden_states", "use_cache"]
    }

    output.update(
        booleans_processing(
            config=config,
            **boolean_dict,
        )
    )

    return output