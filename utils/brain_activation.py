
import os
import mne
import json
import numpy as np
import matplotlib.pyplot as plt

SUBGROUP_SIZE = 4

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
    factor = 4 if len(activation) < 1000 else 1

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

    plt.savefig("results/img_results/{}/{}/{}/EEG/iteration_{}.png".format(model_name, time, n_round, n_round))  # should before show method
    plt.close()

    left = None
    right = None
    vertex = None

    del left
    del right
    del vertex