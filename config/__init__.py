
REGISTRY = {}

from .config_gpt2 import GPT2Config
REGISTRY["GPT2Config"] = GPT2Config

from .config_utils import PretrainedConfig
REGISTRY["PretrainedConfig"] = PretrainedConfig