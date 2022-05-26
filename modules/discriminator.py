
import tensorflow as tf

class tmp(tf.keras.layers.Layer):

    def __init__(self, config,  **kwargs) -> None:
        super().__init__(config, **kwargs)
        
        self.conv_a = tf.keras.layers.Conv2D(64, (5, 5), strides = (2, 2), padding = 'same')
        self.relu_a = tf.keras.layers.LeakyReLU()
        self.dropout_a = tf.keras.layers.Dropout(0.3)

        self.conv_b = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding = 'same')
        self.relu_b = tf.keras.layers.LeakyReLU()
        self.dropout_b = tf.keras.layers.Dropout(0.3)

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1)

    def __call__(self, x, training = False):
        
        x = self.conv_a(x)
        x = self.relu_a(x)
        x = self.dropout_a(x, training = training)

        x = self.conv_b(x)
        x = self.relu_b(x)
        x = self.dropout_b(x, training = training)

        x = self.flatten(x)
        x = self.dense(x)

        return x
        
class Discriminator(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.drop_rate = 0.2

        self.conv_a = tf.keras.layers.Conv2D(8, (1, 64), use_bias = False, activation = 'linear', padding='same', name = 'Spectral_filter')
        self.norm_a = tf.keras.layers.BatchNormalization()
        self.dw_conv = tf.keras.layers.DepthwiseConv2D((2089, 1), use_bias = False, padding='valid', depth_multiplier = 2, activation = 'linear', depthwise_constraint = tf.keras.constraints.MaxNorm(max_value=1), name = 'Spatial_filter')
        self.norm_b = tf.keras.layers.BatchNormalization()
        self.elu_a = tf.keras.layers.ELU()
        self.pooling_a = tf.keras.layers.AveragePooling2D((1, 4))
        self.drop_a = tf.keras.layers.Dropout(self.drop_rate)

        self.s_conv = tf.keras.layers.SeparableConv2D(16, (1, 16), use_bias = False, activation = 'linear', padding = 'same')
        self.norm_c = tf.keras.layers.BatchNormalization()
        self.elu_b = tf.keras.layers.ELU()
        self.pooling_b = tf.keras.layers.AveragePooling2D((1, 4))
        self.drop_b = tf.keras.layers.Dropout(self.drop_rate)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1, activation = 'sigmoid', kernel_constraint = tf.keras.constraints.MaxNorm(0.25))

    def forward(self, x, training = False):

        x = self.conv_a(x)
        x = self.norm_a(x)
        x = self.dw_conv(x)
        x = self.norm_b(x)
        x = self.elu_a(x)
        x = self.pooling_a(x)
        x = self.drop_a(x, training = training)

        x = self.s_conv(x)
        x = self.norm_c(x)
        x = self.elu_b('elu')(x)
        x = self.pooling_b(x)
        x =  self.drop_b(x, training = training)
        x = self.flatten(x)
        x = self.dense(x)

        return x

