
import tensorflow as tf

from model.GPT2GAN import GPT2GAN
from config.config_gpt2 import GPT2Config

config = GPT2Config()
model = GPT2GAN(config = config)

print(model.config)

# defining our optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate = 3e-5, epsilon = 1e-08, clipnorm = 1.0)

# definining our loss function
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

# defining our metric which we want to observe
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

# compiling the model
model.compile(  optimizer = optimizer, 
                loss = [loss, * [None] * model.config.n_layer], 
                metrics = [metric])