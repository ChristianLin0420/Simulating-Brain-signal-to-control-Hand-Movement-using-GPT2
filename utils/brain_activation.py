
import os
import mne
import json
import numpy as np
import matplotlib.pyplot as plt

SUBGROUP_SIZE = 1

def boolean_brain():

    path = '/home/jupyter-ivanljh123/Simulating-Brain-signal-to-control-Hand-Movement-using-GPT2/boolean.json'

    # Read JSON file
    with open(path) as data_file:
        data_loaded = json.load(data_file)

    return (data_loaded["left"], data_loaded["right"])

def transformation_matrix():
    path = '/home/jupyter-ivanljh123/Simulating-Brain-signal-to-control-Hand-Movement-using-GPT2/boolean.json'

    # Read JSON file
    with open(path) as data_file:
        data_loaded = json.load(data_file)

    return data_loaded["transformation"]

def restore_brain_activation(activation, boolean_l, boolean_r):

    l = activation[:255].tolist()
    r = activation[255:].tolist()

    if len(activation) > 1000:
        l = activation[:1022].tolist()
        r = activation[1022:].tolist()

    # print("l length is {}".format(len(l)))
    # print("r length is {}".format(len(r)))

    left_brain_activation = []
    right_brain_activation = []

    zero = [0] * 500

    count = 0
    factor = SUBGROUP_SIZE if len(activation) < 1000 else 1

    # print("factor: {}".format(factor))

    for idx in range(len(boolean_l)):
        if boolean_l[idx]:
            if count % factor == 0 and count < 1020:
                left_brain_activation.append(l[int(count / factor)])
            else:
                left_brain_activation.append(zero)
            count += 1
        else:
            left_brain_activation.append(zero)

    count = 0

    for idx in range(len(boolean_r)):
        if boolean_l[idx]:
            if count % factor == 0 and count < 1064:
                right_brain_activation.append(r[int(count / factor)])
            else:
                right_brain_activation.append(zero)
            count += 1
        else:
            right_brain_activation.append(zero)

    left_brain_activation = np.asarray(left_brain_activation)
    right_brain_activation = np.asarray(right_brain_activation)

    # print("left_brain_activation shape is {}".format(left_brain_activation.shape))
    # print("right_brain_activation shape is {}".format(right_brain_activation.shape))

    return (left_brain_activation, right_brain_activation)

def generate_eeg(real_data, activation_l, activation_r, transformation_matrix, epoch, time, model_name, n_round):
    
    directory1 = 'results/img_results/{}/{}/{}'.format(model_name, time, n_round)
    directory2 = 'results/img_results/{}/{}/{}/EEG'.format(model_name, time, n_round)

    if not os.path.exists(directory1):
        os.mkdir(directory1)
    
    if not os.path.exists(directory2):
        os.mkdir(directory2)

    real = np.asarray(real_data)
    left = np.asarray(activation_l)
    right = np.asarray(activation_r)
    vertex = np.concatenate([left, right], axis = 0)
    t_matrix = np.asarray(transformation_matrix)

    # print("vertex shape is {}".format(vertex.shape))

    real_converted_matrix = np.dot(t_matrix, real)
    fake_converted_matrix = np.dot(t_matrix, vertex)

    # print("converted matrix shape is {}".format(real_converted_matrix.shape))

    title = "Generated EEG Signal(real/fake)"

    # start drawing result
    fig, ax = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    for idx in range(real_converted_matrix.shape[0]):
        ax[0].plot(real_converted_matrix[idx])
        ax[1].plot(fake_converted_matrix[idx])

    plt.savefig("results/img_results/{}/{}/{}/EEG/iteration_{}.png".format(model_name, time, n_round, epoch))  # should before show method
    plt.close()

    left = None
    right = None
    vertex = None

    del left
    del right
    del vertex

def generate_single_channel_eeg_signal(real_data, activation_l, activation_r, transformation_matrix, epoch, time, model_name, n_round):
    
    directory1 = 'results/img_results/{}/{}/{}'.format(model_name, time, n_round)
    directory2 = 'results/img_results/{}/{}/{}/EEG'.format(model_name, time, n_round)

    if not os.path.exists(directory1):
        os.mkdir(directory1)
    
    if not os.path.exists(directory2):
        os.mkdir(directory2)

    real = np.asarray(real_data)
    left = np.asarray(activation_l)
    right = np.asarray(activation_r)
    vertex = np.concatenate([left, right], axis = 0)
    t_matrix = np.asarray(transformation_matrix)

    real_converted_matrix = np.dot(t_matrix, real)
    fake_converted_matrix = np.dot(t_matrix, vertex)

    # start drawing result
    fig, ax = plt.subplots(10, 6, figsize=(48, 20), sharex=True)

    channel_name = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "C3", 
                    "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6", "AFz", "P7", "P3", "Pz", "P4",
                    "P8", "PO9", "O1", "Oz", "O2", "PO10", "AF7", "AF3", "AF4", "AF8", "F5", "F1",
                    "F2", "F6", "FT7", "FC3", "FC4", "FT8", "C1", "C2", "C6", "TP7", "CP3", "CPz",
                    "CP4", "TP8", "P5", "P1", "P2", "P6", "PO7", "PO3", "POz", "PO4", "PO8"]

    for idx in range(real_converted_matrix.shape[0]):
        row = int(idx / 6)
        col = int(idx % 6)
        ax[row, col].set_title(channel_name[idx])
        ax[row, col].plot(real_converted_matrix[idx], label = 'real signal')
        ax[row, col].plot(fake_converted_matrix[idx], label = 'generated signal')

    plt.savefig("results/img_results/{}/{}/{}/EEG/iteration_{:04d}.png".format(model_name, time, n_round, epoch)) 
    plt.close()

    left = None
    right = None
    vertex = None

    del left
    del right
    del vertex

def generate_mne_plot(epoch, time, model_name, n_round):
    directory1 = 'results/img_results/{}/{}/{}'.format(model_name, time, n_round)
    directory2 = 'results/img_results/{}/{}/{}/MNE'.format(model_name, time, n_round)

    if not os.path.exists(directory1):
        os.mkdir(directory1)
    
    if not os.path.exists(directory2):
        os.mkdir(directory2)