
import os

from logging import error
from config.config_gpt2 import load_model_config
from .gpt2gan import gpt2gan
from .gpt2wgan import gpt2wgan



def load_model(model_path, model_name: str = "gpt2gan", noise_len: int = 784, noise_dim: int = 32):

    if not os.path.exists(model_path):
        error("Invalid model path")
    else:
        config = load_model_config(model_path)

        if config is None:
            error("Cannot get model config from given path")
        else:
            if model_name == "gpt2gan":
                model = gpt2gan(config, noise_len, noise_dim)
                model.load_weights(model_path + '.h5')
                return model
            elif model_name == "gpt2wgan":
                model = gpt2wgan(config, noise_len, noise_dim)
                model.load_weights(model_path + '.h5')
                return model
            else:
                error("Given model name is invalid")
                return None

        return None