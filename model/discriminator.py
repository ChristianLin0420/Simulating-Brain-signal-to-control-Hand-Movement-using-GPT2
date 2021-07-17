
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python import keras
from tensorflow.python.ops.gen_array_ops import pad

class Discriminator(tf.keras.layers.Layer):

    def __init__(self, config,  **kwargs) -> None:
        super().__init__(config, **kwargs)

        self.conv2D_a = tf.keras.layers.Conv2D(64, (5, 5), strides = (2, 2), padding = 'same')
        self.relu_a = tf.keras.layers.LeakyReLU()
        self.dropout_a = tf.keras.layers.Dropout(0.3)

        self.conv2D_b = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding = 'same')
        self.relu_b = tf.keras.layers.LeakyReLU()
        self.dropout_b = tf.keras.layers.Dropout(0.3)

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1)
        # self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # def get_loss(self, real_output, fake_output):
    #     real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
    #     fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
    #     total_loss = real_loss + fake_loss
    #     return total_loss

    def __call__(self, x, training=False):
        x = self.conv2D_a(x)
        x = self.relu_a(x)
        x = self.dropout_a(x, training = training)

        x = self.conv2D_b(x)
        x = self.relu_b(x)
        x = self.dropout_b(x, training = training)

        x = self.flatten(x)
        x = self.dense(x)

        return x
        


