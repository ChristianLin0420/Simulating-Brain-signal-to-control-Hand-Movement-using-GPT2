
from threading import local
import tensorflow as tf
from tensorflow.python.keras.engine import input_layer

class MLP(tf.keras.layers.Layer):

    def __init__(   self, 
                    in_feature, 
                    hidden_features = None, 
                    out_features = None, 
                    drop = .0, 
                    **kwargs    ):

        super().__init__(in_feature, hidden_features, out_features, drop, **kwargs)

        out_features = out_features or in_feature
        hidden_features = hidden_features or in_feature
        
        self.f1 = tf.keras.layers.Dense(hidden_features)
        self.act= tf.keras.layers.ReLU()
        self.f2 = tf.keras.layers.Dense(out_features)
        self.drop = tf.keras.layers.Dropout(drop)
        
    def build(self, input_shape):
        pass

    def forward(self, x):
        x = self.f1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.f2(x)
        x = self.drop(x)
        return x

class GPSA(tf.keras.layers.Layer):

    def __init__(   self, 
                    dim, 
                    num_heads = 8, 
                    qkv_bias = False, 
                    qk_scale = None, 
                    attn_drop = 0., 
                    proj_drop = 0.,
                    locality_strength = 1., 
                    use_local_init = True, 
                    **kwargs    ):

        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, locality_strength, use_local_init, **kwargs)

        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim //num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qk = tf.keras.layers.Dense(dim * 2, input_shape = (dim, ), use_bias = qkv_bias)
        self.v = tf.keras.layers.Dense(dim, input_shape = (dim, ), use_bias = qkv_bias)

        self.attn_drop = tf.keras.layers.Dense(attn_drop)
        self.proj = tf.keras.layers.Dense(dim, input_shape = (dim, ))
        self.pos_proj = tf.keras.layers.Dense(num_heads, input_shape = (3, ))
        self.locality_strength = locality_strength
        # self.gating_param = 
        
        if use_local_init:
            self.local_init(locality_strength = locality_strength)


    def local_init(self, locality_strength = 1.):

        self.v.weights.copy(tf.eye(self.dim))
        locality_distance = 1 # max(1, 1 / locality_strength ** .5)

        kernel_size = int(self.num_heads ** .5)
        center = (kernel_size - 1) / 2 if kernel_size % 2 == 0 else kernel_size // 2

        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1 + kernel_size * h2
                self.pos_proj.weights