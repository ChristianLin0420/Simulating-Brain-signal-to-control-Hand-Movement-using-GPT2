
import os
import numpy as np
import tensorflow as tf

from logging import error
from config.config_gpt2 import load_model_config
from .gpt2gan import gpt2gan
from .gpt2wgan import gpt2wgan



def load_model(path, model_name: str = "gpt2gan", noise_len: int = 784, noise_dim: int = 32):

    config = load_model_config(path + '.txt')

    if config is None:
        error("Cannot get model config from given path")
    else:
        if model_name == "gpt2gan":
            model = gpt2gan(config, noise_len, noise_dim)
            model.load_weights(path)
            # weights = model.get_weights()  
            # model.set_weights(weights)
            return model
        elif model_name == "gpt2wgan":
            model = gpt2wgan(config, noise_len, noise_dim)
            model.load_weights(path)
            # weights = model.get_weights()  
            # model.set_weights(weights)
            return model
        else:
            error("Given model name is invalid")
            return None

    return None

# Instance normalization.
def instance_norm(x, epsilon = 1e-8):
    assert len(x.shape) == 3 # NLT

    orig_dtype = x.dtype
    x = tf.cast(x, tf.float32)
    x -= tf.reduce_mean(x, axis = [1, 2], keepdims = True)
    epsilon = tf.constant(epsilon, dtype = x.dtype, name = 'epsilon')
    x *= tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis = [1, 2], keepdims = True) + epsilon)
    x = tf.cast(x, orig_dtype)

    return x

# Upsampling
def upsampling2D(x, size = (2, 1)):
    if x.shape[1] == 2089:
        return x
    else:
        return tf.keras.layers.UpSampling2D(size = size)(x)

# Filled zeros
def filled_zeros(x, size):
    if size != x.shape[1]:
        indicators = np.ones(x.shape)
        indicators[:, size:x.shape[1], :] *= 0
        indicators = tf.convert_to_tensor(indicators)
        indicators = tf.cast(indicators, tf.float32)
        x = tf.math.multiply(x, indicators)
    return x