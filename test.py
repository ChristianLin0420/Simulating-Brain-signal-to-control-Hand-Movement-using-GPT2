
import os
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.python.keras.api._v2.keras import datasets

def initial_mnist_datset(buffer_size: int = 1000, batch_size: int = 8):
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    train_labels = keras.utils.to_categorical(train_labels, 10)

    train_images = np.repeat(train_images, 3, axis = 3)
    
    print(train_images.shape)
    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(buffer_size = buffer_size).batch(batch_size)
    return train_dataset, np.shape(np.asarray(train_dataset))

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

print("----- initial using GPU -----")

# if use_gpu is TRUE, then get one and run the program on it
try:
    gpus = tf.config.list_physical_devices(device_type = 'GPU')
    
    if gpus:
        tf.config.set_visible_devices(devices = gpus[0], device_type = 'GPU')
except:
    print("[No GPR] there is no availible gpu to use!!!")

print("----- start using GPU -----")

print("----- start preparing datasets ------")
datasets, _ = initial_mnist_datset()
print("----- finish preparing datasets ------")

for image, label in datasets:
    # print(image)
    # print(label)
    # print(image)
    # print(tf.image.grayscale_to_rgb(image))
    break
