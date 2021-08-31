
import os
from matplotlib.pyplot import axis
import numpy as np
import tensorflow as tf
import time as tt

from tensorflow import keras

ROOT_DIR = '/home/jupyter-ivanljh123/rsc/Source Estimate'

SUBGROUP_SIZE = 4

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

            # open_data_count = np.load(eye_open_path, allow_pickle = True)
            # close_data_count = np.load(eye_close_path, allow_pickle = True)
            # open_data_count = np.asarray(open_data_count["left"]).shape[0]
            # close_data_count = np.asarray(close_data_count["left"]).shape[0]
            # data_count = min(open_data_count, close_data_count)
            # data_count = data_count / batch_size
            # data_count = data_count * batch_size

            # print("open_data_count: {}, close_data_count: {}, data_count: {}".format(open_data_count, close_data_count, data_count))

    return train_data_filenames, train_data_label

class DatasetGenerator():

    def __init__(self, filenames, labels, batch_size, subject_count) :
        self.image_filenames = filenames
        self.labels = labels

        self.batch_size = batch_size
        self.subjects_count = subject_count

        self.current_subject_index = 0

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
                # left_vertex_count = int(left_shape[1] / SUBGROUP_SIZE)
                # right_vertex_count = int(right_shape[1] / SUBGROUP_SIZE)

                # left_data = left_data[:epoch, :left_vertex_count * SUBGROUP_SIZE, :timestamp]   
                # right_data = right_data[:epoch, :right_vertex_count * SUBGROUP_SIZE, :timestamp]
                left_data = left_data[:epoch, :, :timestamp]   
                right_data = right_data[:epoch, :, :timestamp]
                train_data = np.concatenate((left_data, right_data), axis = 1)
                train_label = np.asarray([batch_y[idx]] * epoch)         

                # for i in range(int(concat_data.shape[1] / SUBGROUP_SIZE)):
                #     if len(train_data) == 0:
                #         train_data = concat_data[:epoch, SUBGROUP_SIZE * i:(SUBGROUP_SIZE * i) + 1, :timestamp]
                #     else:
                #         train_data = np.concatenate((train_data, concat_data[:epoch, SUBGROUP_SIZE * i:(SUBGROUP_SIZE * i) + 1, :timestamp]), axis = 1)
            
            p = np.random.permutation(train_data.shape[0])
            train_data = train_data[p]
            train_label = train_label[p]

            # print("train_data shape: {}".format(train_data.ndim))

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

    ROOT_DIR = '/home/jupyter-ivanljh123/rsc/Source Estimate'

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