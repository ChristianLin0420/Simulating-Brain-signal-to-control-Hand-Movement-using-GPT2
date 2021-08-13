
import tensorflow as tf

class Discriminator(tf.keras.layers.Layer):

    def __init__(self, config,  **kwargs) -> None:
        super().__init__(config, **kwargs)
        
        self.conv_a = tf.keras.layers.Conv2D(64, (5, 5), strides = (2, 2), padding = 'same')
        self.relu_a = tf.keras.layers.LeakyReLU()
        self.dropout_a = tf.keras.layers.Dropout(0.3)

        self.conv_b = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding = 'same')
        self.relu_b = tf.keras.layers.LeakyReLU()
        self.dropout_b = tf.keras.layers.Dropout(0.3)

        # self.conv2D_c = tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding = 'same')
        # self.relu_c = tf.keras.layers.LeakyReLU()
        # self.dropout_c = tf.keras.layers.Dropout(0.3)

        # self.conv2D_d = tf.keras.layers.Conv2D(512, (5, 5), strides=(2, 2), padding = 'same')
        # self.relu_d = tf.keras.layers.LeakyReLU()
        # self.dropout_d = tf.keras.layers.Dropout(0.3)

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1)

    def __call__(self, x, training = False):
        
        print("-" * 100)
        print(x.shape)
        x = self.conv_a(x)
        print(x.shape)
        print("-" * 100)
        x = self.relu_a(x)
        x = self.dropout_a(x, training = training)

        x = self.conv_b(x)
        x = self.relu_b(x)
        x = self.dropout_b(x, training = training)

        # x = self.conv2D_c(x)
        # x = self.relu_c(x)
        # x = self.dropout_c(x, training = training)

        # x = self.conv2D_d(x)
        # x = self.relu_d(x)
        # x = self.dropout_d(x, training = training)

        x = self.flatten(x)
        x = self.dense(x)

        return x
        


