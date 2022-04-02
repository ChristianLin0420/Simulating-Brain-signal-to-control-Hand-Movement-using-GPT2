
# import os
import numpy as np
# import matplotlib.pyplot as plt
# import json 
import tensorflow as tf

input_shape = (1, 2, 2, 3)
x = np.arange(np.prod(input_shape)).reshape(input_shape)
y = tf.keras.layers.UpSampling2D(size=(2, 1))(x)
print(y.shape[:3])