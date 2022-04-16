
import os
import random
from matplotlib.pyplot import axis
import numpy as np
import tensorflow as tf
import time as tt

from tensorflow import keras

ROOT_DIR = '/home/jupyter-ivanljh123/rsc/Source Estimate'

TEMP = 8

def get_training_filenames_and_labels(batch_size: int = 8, subject_count: int = 10):
    _, dirs, files = os.walk(ROOT_DIR).__next__()

    train_data_filenames = []
    train_data_label = []

    dirs = dirs[:subject_count]

    for dir in dirs:
        path = ROOT_DIR + '/' + dir
        _, dirs, files = os.walk(path).__next__()

        if len(files) == 2:
            eye_open_filename = files[0]
            eye_close_filename = files[1]

            eye_open_path = path + '/' + eye_open_filename
            eye_close_path = path + '/' + eye_close_filename

            train_data_filenames.append(eye_open_path)
            train_data_filenames.append(eye_close_path)
            train_data_label.append(1)
            train_data_label.append(0)

    return train_data_filenames, train_data_label

def get_training_raw_signals(subject_count: int = 10):
    _, dirs, files = os.walk('/home/jupyter-ivanljh123/rsc/Source Raw').__next__()

    train_data_filenames = []
    train_data_label = []

    dirs = dirs[:subject_count]

    for dir in dirs:
        path = '/home/jupyter-ivanljh123/rsc/Source Raw' + '/' + dir
        _, dirs, files = os.walk(path).__next__()

        if len(files) == 2:
            eye_open_filename = files[0]
            eye_close_filename = files[1]

            eye_open_path = path + '/' + eye_open_filename
            eye_close_path = path + '/' + eye_close_filename

            train_data_filenames.append(eye_open_path)
            train_data_filenames.append(eye_close_path)
            train_data_label.append(1)
            train_data_label.append(0)

    return train_data_filenames, train_data_label

def get_training_reconstruct_signals():

    _, dirs, files = os.walk('/home/jupyter-ivanljh123/rsc/Source_Reconstructed').__next__()

    subject_folder_name = "A05_hands"

    train_data_filenames = ["/home/jupyter-ivanljh123/rsc/Source_Reconstructed/" + subject_folder_name + "/1_train_X.npz", "/home/jupyter-ivanljh123/rsc/Source_Reconstructed/" + subject_folder_name + "/1_test_X.npz"]
    train_data_label = ["/home/jupyter-ivanljh123/rsc/Source_Reconstructed/" + subject_folder_name + "/1_train_Y.npz", "/home/jupyter-ivanljh123/rsc/Source_Reconstructed/" + subject_folder_name + "/1_test_Y.npz"]

    return train_data_filenames, train_data_label

def generate_random_vectors(num: int = 128, length: int = 2089, emb: int = 500, class_rate_random_vector: float = 0.004, class_count: int = 2, variance: float = 1.0, shuffle: bool = True):

    one_hot_vector_size = int(emb * class_rate_random_vector)

    while(one_hot_vector_size % class_count and num % class_count):
        if one_hot_vector_size % class_count > 0:
            one_hot_vector_size += 1
        if num % class_count > 0:
            num += 1

    random_vectors = np.random.normal(scale = variance, size = (num, length, (emb - one_hot_vector_size)))

    tmp = []
    one_hot = []

    for i in range(class_count):
        t = [i for _ in range(int(num / class_count))]
        tmp += t

    if shuffle:
        random.shuffle(tmp)

    for val in tmp:
        s = []
        sub_vector_size = int(one_hot_vector_size / class_count)

        for i in range(one_hot_vector_size):
            check = int(i / sub_vector_size) == val
            if check:
                s.append(1)
            else:
                s.append(0)

        one_hot.append(s)

    one_hot = np.asarray(one_hot)
    one_hot = np.expand_dims(one_hot, axis = 1)
    one_hot = np.repeat(one_hot, repeats = length, axis = 1)
    random_vectors = np.concatenate([random_vectors, one_hot], axis = 2)
    tmp = np.asarray(tmp)

    return (random_vectors, tmp, sub_vector_size)

def generate_random_vectors_with_labels(labels, num: int = 128, length: int = 2089, emb: int = 500, class_rate_random_vector: float = 0.004, class_count: int = 2, variance: float = 1.0):
    one_hot_vector_size = int(emb * class_rate_random_vector)

    while(one_hot_vector_size % class_count and num % class_count):
        if one_hot_vector_size % class_count > 0:
            one_hot_vector_size += 1
        if num % class_count > 0:
            num += 1

    random_vectors = tf.random.normal(shape = (num, length, (emb - one_hot_vector_size)), stddev = variance)
    sub_vector_size = int(one_hot_vector_size / class_count)

    tmp = labels
    tmp = tf.one_hot(tf.cast(tmp, tf.int32), class_count)
    tmp = tf.reshape(tmp, [-1])
    if sub_vector_size > 1:
        tmp = tf.repeat(tmp, sub_vector_size)
    tmp = tf.reshape(tmp, [one_hot_vector_size, num])
    one_hot = tf.expand_dims(one_hot, axis = 1)
    one_hot = tf.repeat(one_hot, repeats = length, axis = 1)
    random_vectors = tf.concat([random_vectors, one_hot], axis = 2)

    return random_vectors

class DatasetGenerator():

    def __init__(self, filenames, raw_filenames, labels, raw_labels, config = None) :
        self.image_filenames = filenames
        self.raw_filenames = raw_filenames
        self.labels = labels
        self.raw_labels = raw_labels

        self.batch_size = config.batch_size
        self.subjects_count = config.subject_count

        self.current_subject_index = 0
        self.raw_current_subject_index = 0

        self.config = config

    def getItem(self):

        if self.current_subject_index < self.subjects_count:
            
            idx = self.current_subject_index
            
            batch_x = self.image_filenames[idx * 2 : idx * 2 + 1]
            batch_y = self.labels[idx * 2 : idx * 2 + 1]
            
            train_data = np.asarray([])
            train_label = np.asarray([])

            for idx, path in enumerate(batch_x):

                data = np.load(path, allow_pickle = True)

                left_data = data["left"]
                right_data = data["right"]

                left_shape = left_data.shape
                right_shape = right_data.shape

                epoch = min(left_shape[0], right_shape[0])
                epoch = int(epoch / self.batch_size)
                epoch = int(epoch * self.batch_size)

                timestamp = 500 
                left_data = left_data[:epoch, :, :timestamp]   
                right_data = right_data[:epoch, :, :timestamp]
                train_data = np.concatenate((left_data, right_data), axis = 1)
                train_label = np.asarray([batch_y[idx]] * epoch)   
            
            p = np.random.permutation(train_data.shape[0])
            train_data = train_data[p]
            train_label = train_label[p]

            if train_data.ndim != 3:
                self.current_subject_index += 1
                return (-1, -1, False)

            train_data = np.reshape(train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2], 1])
            train_label = keras.utils.to_categorical(train_label, 2)

            train_data = train_data.astype(np.float32)
            train_label = train_label.astype(np.float32)

            self.current_subject_index += 1

            p = None
            del p

            return (train_data, train_label, True)
        else:
            return (None, None, False)

    def get_reconstructed_items(self, X_filenames, Y_filenames):

        train_data = None
        train_label = None

        for x, y in zip(X_filenames, Y_filenames):

            xx = np.load(x, allow_pickle = True)
            yy = np.load(y, allow_pickle = True)

            xx = xx['data']
            yy = yy['data']

            if train_data is None:
                train_data = xx
            else:
                train_data = np.concatenate([train_data, xx], axis = 0)
            
            if train_label is None:
                train_label = yy
            else:
                train_label = np.concatenate([train_label, yy], axis = 0)

        bz = int(train_data.shape[0])

        feet_data = None
        tongue_data = None

        ## get average 
        for data, label in zip(train_data, train_label):
            expand_data = np.asarrar([label])

            if label == 0:
                if feet_data is None:
                    feet_data = expand_data
                else:
                    feet_data = np.concatenate([feet_data, expand_data], axis = 0)
            else:
                if tongue_data is None:
                    tongue_data = expand_data
                else:
                    tongue_data = np.concatenate([tongue_data, expand_data], axis = 0)

        # print("feet data shape: {}".format(feet_data.shape))
        # print("tongue data shape: {}".format(tongue_data.shape))

        feet_data = np.mean(feet_data, axis = 0)
        tongue_data = np.mean(tongue_data, axis = 0)

        # print("feet data shape: {}".format(feet_data.shape))
        # print("tongue data shape: {}".format(tongue_data.shape))

        feet_data = np.expand_dims(feet_data, axis = 0)
        tongue_data = np.expand_dims(tongue_data, axis = 0)

        # print("feet data shape: {}".format(feet_data.shape))
        # print("tongue data shape: {}".format(tongue_data.shape))

        real_data = np.concatenate([feet_data, tongue_data], axis = 0)

        # print("real data shape: {}".format(real_data.shape))

        ## shuffle dataset
        p = np.random.permutation(bz)
        train_data = train_data[p]
        train_label = train_label[p]

        remain_count = int(bz / 8) * 8

        train_data = train_data[:remain_count]
        train_label = train_label[:remain_count]

        train_data = np.reshape(train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2], 1])
        train_label = keras.utils.to_categorical(train_label, self.config.class_count)

        train_data = train_data.astype(np.float32)
        train_label = train_label.astype(np.float32)

        return train_data, train_label, real_data


    def get_event(self):

        if self.current_subject_index < self.subjects_count:
            
            idx = self.current_subject_index
            
            batch_x = self.image_filenames[:TEMP]
            batch_y = self.labels[:TEMP]
            
            train_data = np.asarray([])
            train_label = np.asarray([])

            eye_close_data = np.asarray([])
            eye_open_data = np.asarray([])

            eye_close_exist = False
            eye_open_exist = False

            for idx, path in enumerate(batch_x):

                data = np.load(path, allow_pickle = True)

                left_data = data["left"]
                right_data = data["right"]

                left_shape = left_data.shape
                right_shape = right_data.shape

                epoch = min(left_shape[0], right_shape[0])
                epoch = int(epoch / self.batch_size)
                epoch = int(epoch * self.batch_size)

                timestamp = 500 
                left_data = left_data[:epoch, :, :timestamp]   
                right_data = right_data[:epoch, :, :timestamp]
                train_data = np.concatenate((left_data, right_data), axis = 1)
                train_label = np.asarray([batch_y[idx]] * epoch)   

                if eye_close_exist == False and batch_y[idx] == 0:
                    eye_close_data = train_data[:, :, :]
                    eye_close_exist = True
                if eye_open_exist == False and batch_y[idx] == 1:
                    eye_open_data = train_data[:, :, :]
                    eye_open_exist = True
                if eye_open_exist and eye_close_exist:
                    break

            return (eye_close_data, eye_open_data)
        else:
            return (None, None)


    def get_raw(self):

        if self.raw_current_subject_index < self.subjects_count:
            
            idx = self.raw_current_subject_index
            
            batch_x = self.raw_filenames[:TEMP]
            batch_y = self.raw_labels[:TEMP]
            
            train_data = np.asarray([])
            train_label = np.asarray([])

            eye_close_data = np.asarray([])
            eye_open_data = np.asarray([])

            eye_close_exist = False
            eye_open_exist = False

            for idx, path in enumerate(batch_x):

                data = np.load(path, allow_pickle = True)

                raw_data = data["raw"]
                raw_shape = raw_data.shape

                epoch = raw_shape[0]

                timestamp = 500 
                train_data = raw_data[:epoch, :, :timestamp]   
                train_label = np.asarray([batch_y[idx]] * epoch)   

                if eye_close_exist == False and batch_y[idx] == 0:
                    eye_close_data = train_data[:, :, :]
                    eye_close_exist = True
                if eye_open_exist == False and batch_y[idx] == 1:
                    eye_open_data = train_data[:, :, :]
                    eye_open_exist = True
                if eye_open_exist and eye_close_exist:
                    break

            return (eye_close_data, eye_open_data)
        else:
            return (None, None)




#### testing dataset ####

def initial_mnist_datset(buffer_size: int = 1000, batch_size: int = 8):
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    # train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    train_images = np.repeat(train_images, 3, axis = 3)
    train_labels = keras.utils.to_categorical(train_labels, 10)

    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(buffer_size = buffer_size).batch(batch_size)
    return train_dataset, np.shape(np.asarray(train_dataset))

def dataset_np(last_dim: int = 1):
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    if last_dim == 3:
        train_images = np.repeat(train_images, 3, axis = 3)

    train_labels = keras.utils.to_categorical(train_labels, 10)

    return (train_images, train_labels)

def load_dataset(start_index: int = 0):
    start_time = tt.time()

    _, dirs, files = os.walk(ROOT_DIR).__next__()
    dirs_len = len(dirs)

    train_data = np.asarray([])
    train_label = np.asarray([])

    subject_count = 1 # 4

    if start_index >= dirs_len or (start_index + subject_count) >= dirs_len:
        return None

    dirs = dirs[start_index:(start_index + subject_count)]

    for dir in dirs:
        path = ROOT_DIR + '/' + dir
        print(path)
        _, dirs, files = os.walk(path).__next__()

        if len(files) == 2:
            eye_open_filename = files[0]
            eye_close_filename = files[1]

            eye_open_path = path + '/' + eye_open_filename
            eye_close_path = path + '/' + eye_close_filename

            eye_open_data = np.load(eye_open_path, allow_pickle = True)
            eye_close_data = np.load(eye_close_path, allow_pickle = True)

            left_open_epoch = len(eye_open_data["left"])
            right_open_epoch = len(eye_open_data["right"])
            left_close_epoch = len(eye_close_data["left"])
            right_close_epoch = len(eye_close_data["right"])

            open_epoch = min(left_open_epoch, right_open_epoch)
            close_epoch = min(left_close_epoch, right_close_epoch)
            vertex_count = 1022
            timestemp = 500

            left_open_data = np.asarray(eye_open_data["left"][:open_epoch, :vertex_count, :timestemp])
            right_open_data = np.asarray(eye_open_data["right"][:open_epoch, :vertex_count, :timestemp])
            left_close_data = np.asarray(eye_close_data["left"][:close_epoch, :vertex_count, :timestemp])
            right_close_data = np.asarray(eye_close_data["right"][:close_epoch, :vertex_count, :timestemp])

            eye_open_data = np.concatenate((left_open_data, right_open_data), axis = 1)
            eye_close_data = np.concatenate((left_close_data, right_close_data), axis = 1)

            if len(train_data) == 0 and len(train_label) == 0:
                train_data = eye_open_data
                train_label = np.asarray([1] * open_epoch) 
                train_data = np.concatenate((train_data, eye_close_data), axis = 0)
                train_label = np.concatenate((train_label, np.asarray([0] * close_epoch)), axis = 0)
            else:
                assert train_data.shape[0] == train_label.shape[0]

                train_data = np.concatenate((train_data, eye_open_data), axis = 0)
                train_label = np.concatenate((train_label, np.asarray([1] * open_epoch)), axis = 0)
                train_data = np.concatenate((train_data, eye_close_data), axis = 0)
                train_label = np.concatenate((train_label, np.asarray([0] * close_epoch)), axis = 0)
    
    assert train_data.shape[0] == train_label.shape[0]

    p = np.random.permutation(train_data.shape[0])
    train_data = train_data[p]
    train_label = train_label[p]

    train_data = np.reshape(train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2], 1])
    train_label = keras.utils.to_categorical(train_label, 2)

    train_data = train_data.astype(np.float32)
    train_label = train_label.astype(np.float32)

    print("-" * 100)
    print(train_data.shape)
    print(train_label.shape)

    data_count = int(train_data.shape[0] / 8)
    data_count = data_count * 8

    print("--- %s seconds ---" % (tt.time() - start_time))

    return (train_data[:data_count], train_label[:data_count])