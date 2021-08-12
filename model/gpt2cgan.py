
import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import real

from .gpt2 import TFGPT2MainLayer
from .discriminator import Discriminator


class gpt2cgan(tf.keras.Model):

    def __init__(self, config, noise_len: int = 784, noise_dim: int = 32, d_extra_steps: int = 5, last_dim: int = 3, **kwargs):
        super(gpt2cgan, self).__init__()

        self.generator = TFGPT2MainLayer(config = config, name = "name", last_dim = last_dim)
        self.discriminator = Discriminator(config = config, name = "discriminator")

        self.noise_len = noise_len
        self.noise_dim = noise_dim

        self.gp_weight = 10.0
        self.d_extra_steps = d_extra_steps

        # add one hot vector for every seed
        self.seed = tf.random.normal([16, noise_len, noise_dim])
        l = tf.constant([x % 10 for x in range(16)])
        one_hot = tf.one_hot(l, depth = 10)
        one_hot = tf.expand_dims(one_hot, axis = 1)
        one_hot = tf.repeat(one_hot, repeats = noise_len, axis = 1)
        self.seed = tf.concat([self.seed, one_hot], axis = 2)

        self.last_dim = last_dim

        self.config = config

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(gpt2cgan, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def build(self, input_shape):
        return super().build(input_shape)

    def gradient_penalty(self, 
                         real_images, 
                         fake_images, 
                         batch_size: int = 8):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training = True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]

        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis = [1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, datasets):

        # real_images = tf.constant(0)
        # real_labels = tf.constant(0)

        # print("dataset: {}".format(datasets))

        real_images, real_labels = datasets

        # for image, label in datasets:
        #     real_images = image
        #     real_labels = label

        if self.last_dim > 1:
            real_images = tf.image.grayscale_to_rgb(real_images)

        # print(real_images)
        # print(real_labels)
        # print(type(real_images))
        # print(type(real_labels))

        # one hot information
        one_hot_labels = tf.expand_dims(real_labels, axis = 1)
        one_hot_labels = tf.repeat(one_hot_labels, repeats = self.noise_len, axis = 1)
        
        # Sample random points in the latent space
        batch_size = 8
        # print("*" * 100)
        # print(type(batch_size))
        # print(batch_size)
        d_loss = 0

        for _ in range(self.d_extra_steps):
            random_latent_vectors = tf.random.normal(shape = (batch_size, self.noise_len, self.noise_dim))

            random_latent_vectors = tf.concat([random_latent_vectors, one_hot_labels], axis = 2)

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
                # Calculate the gradient penalty
                gp = self.gradient_penalty(real_images, generated_images, batch_size)
                # Add the gradient penalty to the original discriminator loss
                d_loss = self.loss_fn(labels, predictions) + gp * self.gp_weight

            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape = (batch_size, self.noise_len, self.noise_dim))
        random_latent_vectors = tf.concat([random_latent_vectors, one_hot_labels], axis = 2)
        
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
        
        return {"d_loss": d_loss / self.d_extra_steps, "g_loss": g_loss, "predictions": predictions}
