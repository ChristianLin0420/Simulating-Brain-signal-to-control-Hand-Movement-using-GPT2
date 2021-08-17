

# import os
# import gdown
# import tarfile

# from numpy.random.mtrand import rand

# output = 'test.npz'

# def extract(tar_url, extract_path='.'):
#     tar = tarfile.open(output, 'r')
#     for item in tar:
#         tar.extract(item, extract_path)
#         if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
#             extract(item.name, "./" + item.name[:item.name.rfind('/')])

# if not os.path.exists(output):
#     url = 'https://drive.google.com/file/d/1-WyVlbk7njGie4FK6iM_ghuPcRHIrxCg/view?usp=sharing'
#     gdown.download(url, output, quiet=False)
# else:
#     extract(output)

# import numpy as np
# import matplotlib.pyplot as plt
# data = np.load('/Users/christianlin/Desktop/res/010002/010002_EC.npz', allow_pickle = True)

# print(data["left"].shape)
# print(data["right"].shape)
# print(data['left'][0][0][:10])

# random = np.random.random([28, 28, 3])
# max = np.max(random)
# min = np.min(random)

# print(random)

# random = (random - min) / (max - min)
# plt.imshow(random)
# plt.show()


import tensorflow as tf
import numpy as np
from tensorflow import keras

import matplotlib.pyplot as plt
from tensorflow.python.ops.gen_math_ops import imag


def dataset_np(last_dim: int = 1):
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    # train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    if last_dim == 3:
        train_images = np.repeat(train_images, 3, axis = 3)

    train_labels = keras.utils.to_categorical(train_labels, 10)

    return (train_images, train_labels)


dataset = dataset_np(3)

image = dataset[0][1]
label = dataset[1][:5]

# print(label)
# one_hot_labels = np.expand_dims(label, axis = 1)
# print(one_hot_labels)
# one_hot_labels = np.repeat(one_hot_labels, repeats = 10, axis = 1)
# print(one_hot_labels)

# image_size = 28
# num_classes = 10

# image_one_hot_labels = label[:, :, None, None]
# print(image_one_hot_labels)
# image_one_hot_labels = tf.repeat(
#     image_one_hot_labels, repeats=[image_size * image_size]
# )
# print(image_one_hot_labels)
# image_one_hot_labels = tf.reshape(
#     image_one_hot_labels, (-1, image_size, image_size, num_classes)
# )
# print(image_one_hot_labels)
# print(image)
# print(np.max(image))
# print(np.min(image))


# plt.imshow(image)
# plt.show()


l = [-0.1, 0, 0.9, 1, 1.2]
n = np.asarray(l)
print(n)
n = n if n <= 1.0 else 1.0

print(n)































# import os
# import numpy as np
# import tensorflow as tf

# from tensorflow import keras
# from tensorflow.python.keras.api._v2.keras import datasets

# def initial_mnist_datset(buffer_size: int = 1000, batch_size: int = 8):
#     (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
#     train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
#     train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
#     train_labels = keras.utils.to_categorical(train_labels, 10)

#     train_images = np.repeat(train_images, 3, axis = 3)
    
#     print(train_images.shape)
#     # Batch and shuffle the data
#     train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(buffer_size = buffer_size).batch(batch_size)
#     return train_dataset, np.shape(np.asarray(train_dataset))

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# print("----- initial using GPU -----")

# # if use_gpu is TRUE, then get one and run the program on it
# try:
#     gpus = tf.config.list_physical_devices(device_type = 'GPU')
    
#     if gpus:
#         tf.config.set_visible_devices(devices = gpus[0], device_type = 'GPU')
# except:
#     print("[No GPR] there is no availible gpu to use!!!")

# print("----- start using GPU -----")

# print("----- start preparing datasets ------")
# datasets, _ = initial_mnist_datset()
# print("----- finish preparing datasets ------")

# for image, label in datasets:
#     # print(image)
#     # print(label)
#     # print(image)
#     # print(tf.image.grayscale_to_rgb(image))
#     break
