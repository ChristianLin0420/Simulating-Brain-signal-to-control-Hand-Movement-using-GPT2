

import os
import time
import numpy as np

from sys import getsizeof
from tensorflow import keras


start_time = time.time()

ROOT_DIR = '/home/jupyter-ivanljh123/rsc/Source Estimate'

subdirs, dirs, files = os.walk(ROOT_DIR).__next__()

train_data = np.asarray([])
train_label = np.asarray([])

dirs = dirs[:5]

for dir in dirs:
    path = ROOT_DIR + '/' + dir
    subdirs, dirs, files = os.walk(path).__next__()

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

data_count = int(train_data.shape[0] / 8)
data_count = data_count * 8

train_data = train_data[:data_count]
train_label = train_label[:data_count]

train_data = np.reshape(train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2], 1])
train_label = keras.utils.to_categorical(train_label, 2)

print("train data size: {}".format(getsizeof(train_data)))
print("train label size: {}".format(getsizeof(train_label)))

print("--- %s seconds ---" % (time.time() - start_time))

