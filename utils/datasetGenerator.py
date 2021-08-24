
import os
from matplotlib.pyplot import axis
import numpy as np
import tensorflow as tf

from tensorflow import keras

ROOT_DIR = '/home/jupyter-ivanljh123/rsc/Source Estimate'

SUBGROUP_SIZE = 4

def get_training_filenames_and_labels(subject_count: int = 10):
    _, dirs, files = os.walk(ROOT_DIR).__next__()

    train_data_filenames = []
    train_data_label = []

    dirs = dirs[:subject_count]
    steps = 0

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

            open_data_count = np.load(eye_open_path, allow_pickle = True).shape[0]
            close_data_count = np.load(eye_close_path, allow_pickle = True).shape[0]
            data_count = min(open_data_count, close_data_count)
            data_count = data_count / SUBGROUP_SIZE
            data_count = data_count * SUBGROUP_SIZE
            steps += (data_count * 2)

            print("open_data_count: {}, close_data_count: {}, data_count: {}, steps: {}".format(open_data_count, close_data_count, data_count, steps))

    return train_data_filenames, train_data_label, steps

class DatasetGenerator(keras.utils.Sequence):

    def __init__(self, filenames, labels, batch_size, subject_count) :
        self.image_filenames = filenames
        self.labels = labels

        self.current_train_data = np.asarray([])
        self.current_train_label = np.asarray([])

        self.batch_size = batch_size
        self.subjects_count = subject_count

        self.current_subject_index = 0
        self.current_start_index = 0
        self.current_train_batch_count = -1
        
    def __len__(self) :
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, index):

        print("current_train_batch_count: {}, current_start_index: {}, current_subject_index: {}".format(self.current_train_batch_count, self.current_start_index, self.current_subject_index))

        if self.current_train_batch_count == -1 or self.current_start_index == 0 and self.current_subject_index < self.subjects_count:
            
            print("============= Generating new train dataset =============")

            idx = self.current_subject_index
            
            batch_x = self.image_filenames[idx * self.batch_size : (idx + 1) * self.batch_size]
            batch_y = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]
            
            train_data = np.asarray([])
            train_label = np.asarray([])

            for idx, path in enumerate(batch_x):
                
                tmp_train_data = np.asarray([])
                tmp_train_label = np.asarray([])

                data = np.load(path, allow_pickle = True)

                left_data = data["left"]
                right_data = data["right"]

                left_shape = left_data.shape
                right_shape = right_data.shape

                epoch = min(left_shape[0], right_shape[0])
                epoch = epoch / SUBGROUP_SIZE
                epoch = epoch * SUBGROUP_SIZE

                timestamp = left_shape[-1]
                left_vertex_count = left_shape[1] / SUBGROUP_SIZE
                right_vertex_count = right_shape[1] / SUBGROUP_SIZE
                vertex_count = min(left_vertex_count, right_vertex_count)

                left_data = left_data[:epoch, :vertex_count * SUBGROUP_SIZE, :timestamp]   
                right_data = right_data[:epoch, :vertex_count * SUBGROUP_SIZE, :timestamp]
                concat_data = np.concatenate((left_data, right_data), axis = 1)
                label = np.asarray([batch_y[idx]] * epoch)         

                for i in range(left_vertex_count):
                    if len(tmp_train_data) == 0 and len(tmp_train_label) == 0:
                        tmp_train_data = concat_data[:epoch, SUBGROUP_SIZE * i, :timestamp]
                        tmp_train_label = label
                    else:
                        tmp_train_data = np.concatenate((tmp_train_data, concat_data[:epoch, SUBGROUP_SIZE * i, :timestamp]), axis = 1)
                        tmp_train_label = np.concatenate((tmp_train_label, label), axis = 0)

                if len(train_data) == 0 and len(train_label) == 0:
                    train_data = tmp_train_data
                    train_label = tmp_train_label
                else:
                    assert train_data.shape[0] == train_label.shape[0]

                    train_data = np.concatenate((train_data, tmp_train_data), axis = 0)
                    train_label = np.concatenate((train_label, tmp_train_label), axis = 0)

                tmp_train_data = None
                tmp_train_label = None

                del tmp_train_data
                del tmp_train_label
            
            p = np.random.permutation(train_data.shape[0])
            train_data = train_data[p]
            train_label = train_label[p]

            train_data = np.reshape(train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2], 1])
            train_label = keras.utils.to_categorical(train_label, 2)

            self.current_train_data = train_data.astype(np.float32)
            self.current_train_label = train_label.astype(np.float32)

            self.current_start_index = 1

            train_data = None
            train_label = None
            p = None

            del train_data
            del train_label
            del p

            return (self.current_train_data[:SUBGROUP_SIZE], self.current_train_label[:SUBGROUP_SIZE])

        elif self.current_start_index < self.current_train_batch_count and self.current_subject_index < self.subjects_count :

            train_data = self.current_train_data[self.current_start_index * SUBGROUP_SIZE : (self.current_start_index + 1) * SUBGROUP_SIZE]
            train_label = self.current_train_label[self.current_start_index * SUBGROUP_SIZE : (self.current_start_index + 1) * SUBGROUP_SIZE]
            
            self.current_start_index += 1

            if self.current_start_index == self.current_train_batch_count:
                self.current_start_index = 0
                self.current_subject_index += 1

            return (train_data, train_label)
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