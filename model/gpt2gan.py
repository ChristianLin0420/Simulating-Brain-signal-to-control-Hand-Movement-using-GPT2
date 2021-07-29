


import tensorflow as tf

from datetime import date, datetime

from .gpt2 import TFGPT2MainLayer
from .discriminator import Discriminator


class gpt2gan(tf.keras.Model):

    def __init__(self, config, noise_len: int = 784, noise_dim: int = 32, **kwargs):
        super(gpt2gan, self).__init__()

        self.generator = TFGPT2MainLayer(config = config, name = "name")
        self.discriminator = Discriminator(config = config, name = "discriminator")

        self.noise_len = noise_len
        self.noise_dim = noise_dim
        self.seed = tf.random.normal([16, noise_len, noise_dim])

        self.config = config

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(gpt2gan, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.noise_len, self.noise_dim))

        # generate images from gpt2 
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)
        
        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the disciminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.noise_len, self.noise_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # generate image from given seed
        predictions = self.generator(self.seed, training = False)
        
        
        return {"d_loss": d_loss, "g_loss": g_loss, "predictions": predictions}
