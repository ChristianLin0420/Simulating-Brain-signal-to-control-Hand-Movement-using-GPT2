
# from .transformers.models.gpt2.modeling_tf_gpt2 import TFGPT2MainLayer
from transformers import GPT2Config, TFGPT2MainLayer

config = GPT2Config(vocab_size=50257, 
                    n_positions=1024, 
                    n_ctx=1024, 
                    n_embd=768, 
                    n_layer=12, 
                    n_head=12, 
                    n_inner=None, 
                    activation_function="gelu_new", 
                    resid_pdrop=0.1, 
                    embd_pdrop=0.1, 
                    attn_pdrop=0.1, 
                    layer_norm_epsilon=0.00001, 
                    initializer_range=0.02)

generator = TFGPT2MainLayer(config = config)

print(generator.config)