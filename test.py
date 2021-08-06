

from numpy import float32
from numpy.random.mtrand import random
import tensorflow as tf
from tensorflow import keras
import numpy as np

def initial_mnist_datset(buffer_size: int = 1000, batch_size: int = 8):
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    train_labels = keras.utils.to_categorical(train_labels, 10)

    tmp = tf.image.grayscale_to_rgb(tf.constant(train_images))
    print(tmp.shape)

    print(train_images.shape)
    print(train_labels.shape)

    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(buffer_size = buffer_size).batch(batch_size)
    return train_dataset, np.shape(np.asarray(train_dataset))

dataset, _ = initial_mnist_datset()

# images, labels = dataset

# print(dataset)

random_latent_vectors = tf.random.normal(shape = (2, 5, 4))

print("original random latent vectors:")
print(random_latent_vectors)

one_hot_labels = tf.constant([[0, 0, 0, 1], [1, 0, 0, 0]], dtype = float32)
print(one_hot_labels.shape)
one_hot_labels = tf.expand_dims(one_hot_labels, axis = 1)
one_hot_labels = tf.repeat(one_hot_labels, repeats = 5, axis = 1)
# one_hot_labels = tf.repeat(one_hot_labels, repeats = 5)
# print(one_hot_labels.shape)
# one_hot_labels = tf.reshape(one_hot_labels, shape = [-1, 5, 4])

print(one_hot_labels.shape)

random_latent_vectors = tf.concat(
    [random_latent_vectors, one_hot_labels], axis = 2
)

print("modified latent vectors:")
print(random_latent_vectors)