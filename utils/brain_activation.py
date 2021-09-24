
import os
import mne
import json
import os.path as op
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from numpy.fft import fft

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

def restore_brain_activation_tf(activation, boolean_l, boolean_r):
    
    l = activation[:255]
    r = activation[255:]

    if len(activation) > 1000:
        l = activation[:1022]
        r = activation[1022:]

    print("l shape: {}".format(l.shape))
    print("r shape: {}".format(r.shape))

    left_brain_activation = tf.constant([])
    right_brain_activation = tf.constant([])

    zero = [0] * 500
    zero = tf.constant(zero)
    zero = tf.expand_dims(zero, axis = 0)

    count = 0
    factor = SUBGROUP_SIZE if len(activation) < 1000 else 1

    for idx in range(len(boolean_l)):
        if boolean_l[idx]:
            if count % factor == 0 and count < 1020:
                tmp = l[int(count / factor)]
                tmp = tf.expand_dims(tmp, axis = 0)

                if len(left_brain_activation) == 0:
                    left_brain_activation = tmp
                else:
                    left_brain_activation = tf.concat([left_brain_activation, tmp], axis = 0)
            else:
                left_brain_activation = tf.concat([left_brain_activation, zero], axis = 0)
            count += 1
        else:
            left_brain_activation = tf.concat([left_brain_activation, zero], axis = 0)

    count = 0

    for idx in range(len(boolean_r)):
        if boolean_l[idx]:
            if count % factor == 0 and count < 1064:
                tmp = l[int(count / factor)]
                tmp = tf.expand_dims(tmp, axis = 0)

                if len(left_brain_activation) == 0:
                    right_brain_activation = tmp
                else:
                    right_brain_activation = tf.concat([right_brain_activation, tmp], axis = 0)
            else:
                right_brain_activation = tf.concat([right_brain_activation, zero], axis = 0)
            count += 1
        else:
            right_brain_activation = tf.concat([right_brain_activation, zero], axis = 0)

    print("left_brain_activation shape: {}".format(left_brain_activation.shape))
    print("right_brain_activation shape: {}".format(right_brain_activation.shape))

    return (left_brain_activation, right_brain_activation)

def restore_brain_activation(activation, boolean_l, boolean_r):

    l = activation[:255].tolist()
    r = activation[255:].tolist()

    if len(activation) > 1000:
        l = activation[:1022].tolist()
        r = activation[1022:].tolist()

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

def fetch_brain_template():

    data_path = "/home/jupyter-ivanljh123/rsc/EEG_Preprocessed"

    input_fname = op.join(data_path, 'sub-010286_EC.set')
        
    # Load preprocessed data
    raw = mne.io.read_raw_eeglab(input_fname, preload=True, verbose=False)

    # Set montage
    # Read and set the EEG electrode locations
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)

    # Set common average reference
    raw.set_eeg_reference('average', projection=True, verbose=False)

    # Construct epochs
    events, _ = mne.events_from_annotations(raw, verbose=False)
    raw.info["events"] = events

    event_id = {"eyes close": 2}

    tmin, tmax = 0., 2.  # in s
    baseline = None

    epochs = mne.Epochs(raw, 
                        events=events,
                        event_id=event_id, 
                        tmin=tmin,
                        tmax=tmax, 
                        baseline=baseline, 
                        verbose=False ) 

    # plot evoked response for face A
    evoked = epochs.average().pick('eeg')

    del raw

    return evoked

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

def generate_single_channel_eeg_signal(raw_close_data, raw_open_data, real_close_data, real_open_data, close_activation_l, close_activation_r, open_activation_l, open_activation_r, transformation_matrix, epoch, time, model_name, n_round, event_idx):
    
    directory1 = 'results/img_results/{}/{}/{}'.format(model_name, time, n_round)
    directory2 = 'results/img_results/{}/{}/{}/EEG'.format(model_name, time, n_round)
    directory3 = 'results/img_results/{}/{}/{}/EEG/iteration_{:04d}'.format(model_name, time, n_round, epoch)

    if not os.path.exists(directory1):
        os.mkdir(directory1)
    
    if not os.path.exists(directory2):
        os.mkdir(directory2)

    if not os.path.exists(directory3):
        os.mkdir(directory3)

    channel_name = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "C3", 
                    "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6", "AFz", "P7", "P3", "Pz", "P4",
                    "P8", "PO9", "O1", "Oz", "O2", "PO10", "AF7", "AF3", "AF4", "AF8", "F5", "F1",
                    "F2", "F6", "FT7", "FC3", "FC4", "FT8", "C1", "C2", "C6", "TP7", "CP3", "CPz",
                    "CP4", "TP8", "P5", "P1", "P2", "P6", "PO7", "PO3", "POz", "PO4", "PO8"]

    real_close_data_tmp = np.asarray(real_close_data[0])
    left = np.asarray(close_activation_l[0])
    right = np.asarray(close_activation_r[0])
    close_vertex = np.concatenate([left, right], axis = 0)
    t_matrix = np.asarray(transformation_matrix)

    real_close_converted_matrix = np.dot(t_matrix, real_close_data_tmp)
    generated_close_converted_matrix = np.dot(t_matrix, close_vertex)

    # start drawing result
    fig, ax = plt.subplots(10, 6, figsize=(48, 30), sharex=True)

    for idx in range(real_close_converted_matrix.shape[0]):
        row = int(idx / 6)
        col = int(idx % 6)
        ax[row, col].set_title(channel_name[idx])
        ax[row, col].plot(real_close_converted_matrix[idx], label = 'real signal')
        ax[row, col].plot(generated_close_converted_matrix[idx], label = 'generated signal')

    plt.savefig("results/img_results/{}/{}/{}/EEG/iteration_{:04d}/Eye_Close_{:04d}.png".format(model_name, time, n_round, epoch, event_idx)) 
    plt.close()

    real_open_data_tmp = np.asarray(real_open_data[0])
    left = np.asarray(open_activation_l[0])
    right = np.asarray(open_activation_r[0])
    open_vertex = np.concatenate([left, right], axis = 0)
    t_matrix = np.asarray(transformation_matrix)

    real_open_converted_matrix = np.dot(t_matrix, real_open_data_tmp)
    generated_open_converted_matrix = np.dot(t_matrix, open_vertex)

    # start drawing result
    fig, ax = plt.subplots(10, 6, figsize=(48, 30), sharex=True)

    for idx in range(real_open_converted_matrix.shape[0]):
        row = int(idx / 6)
        col = int(idx % 6)
        ax[row, col].set_title(channel_name[idx])
        ax[row, col].plot(real_open_converted_matrix[idx], label = 'real signal')
        ax[row, col].plot(generated_open_converted_matrix[idx], label = 'generated signal')

    plt.savefig("results/img_results/{}/{}/{}/EEG/iteration_{:04d}/Eye_Open_{:04d}.png".format(model_name, time, n_round, epoch, event_idx)) 
    plt.close()

    target_channel = ["Oz"] #["C3", "C4", "Cz", "O1", "O2", "Oz"]

    # apply sftf
    # for idx in range(real_close_converted_matrix.shape[0]):
    #     if channel_name[idx] in target_channel:
    #         generate_power_spectrum(False, channel_name[idx], real_close_converted_matrix[idx], generated_close_converted_matrix[idx], epoch, time, model_name, n_round, event_idx)

    # for idx in range(real_open_converted_matrix.shape[0]):
    #     if channel_name[idx] in target_channel:
    #         generate_power_spectrum(True, channel_name[idx], real_open_converted_matrix[idx], generated_open_converted_matrix[idx], epoch, time, model_name, n_round, event_idx)

    for idx in range(len(channel_name)):
        if channel_name[idx] in target_channel:
            generate_stft(False, raw_close_data, real_close_data, close_activation_l, close_activation_r, transformation_matrix, epoch, time, model_name, n_round, channel_name[idx], idx)
            generate_stft(True, raw_open_data, real_open_data, open_activation_l, open_activation_r, transformation_matrix, epoch, time, model_name, n_round, channel_name[idx], idx)


    real_close_data = None
    real_close_converted_matrix = None
    generated_close_converted_matrix = None
    real_open_data = None
    real_open_converted_matrix = None
    generated_open_converted_matrix = None
    left = None
    right = None
    close_vertex = None
    open_vertex = None
    t_matrix = None

    del real_close_data
    del real_close_converted_matrix
    del generated_close_converted_matrix
    del real_open_data
    del real_open_converted_matrix
    del generated_open_converted_matrix
    del left
    del right
    del close_vertex
    del open_vertex
    del t_matrix

def generate_mne_plot(brain_template, real_close_data, real_open_data, close_activation_l, close_activation_r, open_activation_l, open_activation_r, transformation_matrix, epoch, time, model_name, n_round, event_idx):

    directory1 = 'results/img_results/{}/{}/{}'.format(model_name, time, n_round)
    directory2 = 'results/img_results/{}/{}/{}/MNE'.format(model_name, time, n_round)
    directory3 = 'results/img_results/{}/{}/{}/MNE/iteration_{:04d}'.format(model_name, time, n_round, epoch)
    directory4 = 'results/img_results/{}/{}/{}/MNE/iteration_{:04d}/Eye_Close'.format(model_name, time, n_round, epoch)
    directory5 = 'results/img_results/{}/{}/{}/MNE/iteration_{:04d}/Eye_Open'.format(model_name, time, n_round, epoch)

    if not os.path.exists(directory1):
        os.mkdir(directory1)
    
    if not os.path.exists(directory2):
        os.mkdir(directory2)

    if not os.path.exists(directory3):
        os.mkdir(directory3)

    if not os.path.exists(directory4):
        os.mkdir(directory4)

    if not os.path.exists(directory5):
        os.mkdir(directory5)

    real_close_data = np.asarray(real_close_data[0])
    left = np.asarray(close_activation_l[0])
    right = np.asarray(close_activation_r[0])
    close_vertex = np.concatenate([left, right], axis = 0)
    t_matrix = np.asarray(transformation_matrix)

    real_close_converted_matrix = np.dot(t_matrix, real_close_data)
    generated_close_converted_matrix = np.dot(t_matrix, close_vertex)

    # plot brain activation
    brain_template.data = real_close_converted_matrix

    ax = brain_template.plot_topomap(times=np.linspace(0.0, 0.2, 20), ch_type='eeg', time_unit='s', ncols=5, nrows='auto', title = 'Original Eye Close Brain Activation in iteration {}'.format(epoch), show = False)
    ax.savefig("results/img_results/{}/{}/{}/MNE/Eye_Close_Original.png".format(model_name, time, n_round))
    plt.close(ax)

    brain_template.data = generated_close_converted_matrix

    ax = brain_template.plot_topomap(times=np.linspace(0.0, 0.2, 20), ch_type='eeg', time_unit='s', ncols=5, nrows='auto', title = 'Generated Eye Close Brain Activation in iteration {}'.format(epoch), show = False)
    ax.savefig("results/img_results/{}/{}/{}/MNE/iteration_{:04d}/Eye_Close/Generated_{:04d}.png".format(model_name, time, n_round, epoch, event_idx))
    plt.close(ax)


    real_open_data = np.asarray(real_open_data[0])
    left = np.asarray(open_activation_l[0])
    right = np.asarray(open_activation_r[0])
    open_vertex = np.concatenate([left, right], axis = 0)
    t_matrix = np.asarray(transformation_matrix)

    real_open_converted_matrix = np.dot(t_matrix, real_open_data)
    generated_open_converted_matrix = np.dot(t_matrix, open_vertex)

    # plot brain activation
    brain_template.data = real_open_converted_matrix

    ax = brain_template.plot_topomap(times=np.linspace(0.0, 0.2, 20), ch_type='eeg', time_unit='s', ncols=5, nrows='auto', title = 'Original Eye Open Brain Activation in iteration {}'.format(epoch), show = False)
    ax.savefig("results/img_results/{}/{}/{}/MNE/Eye_Open_Original.png".format(model_name, time, n_round))
    plt.close(ax)

    brain_template.data = generated_open_converted_matrix

    ax = brain_template.plot_topomap(times=np.linspace(0.0, 0.2, 20), ch_type='eeg', time_unit='s', ncols=5, nrows='auto', title = 'Generated Eye Open Brain Activation in iteration {}'.format(epoch), show = False)
    ax.savefig("results/img_results/{}/{}/{}/MNE/iteration_{:04d}/Eye_Open/Generated_{:04d}.png".format(model_name, time, n_round, epoch, event_idx))
    plt.close(ax)

    real_close_data = None
    real_close_converted_matrix = None
    generated_close_converted_matrix = None
    real_open_data = None
    real_open_converted_matrix = None
    generated_open_converted_matrix = None
    left = None
    right = None
    close_vertex = None
    open_vertex = None
    t_matrix = None

    del real_close_data
    del real_close_converted_matrix
    del generated_close_converted_matrix
    del real_open_data
    del real_open_converted_matrix
    del generated_open_converted_matrix
    del left
    del right
    del close_vertex
    del open_vertex
    del t_matrix

def generate_stft(eye_open, raw_data, process_data, activation_l, activation_r, transformation_matrix, epoch, time, model_name, n_round, channel, channel_idx):
    
    directory1 = 'results/img_results/{}/{}/{}'.format(model_name, time, n_round)
    directory2 = 'results/img_results/{}/{}/{}/Spectrum'.format(model_name, time, n_round)
    directory3 = 'results/img_results/{}/{}/{}/Spectrum/iteration_{:04d}'.format(model_name, time, n_round, epoch)

    if not os.path.exists(directory1):
        os.mkdir(directory1)
    
    if not os.path.exists(directory2):
        os.mkdir(directory2)

    if not os.path.exists(directory3):
        os.mkdir(directory3)

    # plot original signal
    N = 500
    dt = 0.004
    T = 2.0

    fNQ = 1 / dt / 2                            # Determine Nyquist frequency
    faxis = np.arange(0, fNQ, 0.5)              # Construct frequency axis

    t_matrix = np.asarray(transformation_matrix)

    raw_collection = []
    process_collection = []

    raw_average = np.asarray([])
    process_average = np.asarray([])
    generated_average = np.asarray([])

    for idx in range(len(raw_data)):
        signal = np.asarray(raw_data[idx, channel_idx, :]).flatten()
        xf_original = fft(signal - np.mean(signal))
        Sxx_original = 2 * dt ** 2 / T * (xf_original * xf_original.conj())         # Compute spectrum
        Sxx_original = Sxx_original[:int(N / 2)]      
        
        raw_collection.append(Sxx_original)                                         # Ignore negative frequencies

        if len(raw_average) == 0:
            raw_average = Sxx_original
        else:
            raw_average = np.add(raw_average, Sxx_original)

    raw_average = np.true_divide(raw_average, float(len(raw_data)))

    for idx in range(len(process_data)):

        real_open_converted_matrix = np.dot(t_matrix, process_data[idx])

        signal = np.asarray(real_open_converted_matrix[channel_idx]).flatten()
        xf_original = fft(signal - np.mean(signal))
        Sxx_original = 2 * dt ** 2 / T * (xf_original * xf_original.conj())         # Compute spectrum
        Sxx_original = Sxx_original[:int(N / 2)]                                    # Ignore negative frequencies

        process_collection.append(Sxx_original)

        if len(process_average) == 0:
            process_average = Sxx_original
        else:
            process_average = np.add(process_average, Sxx_original)

    process_average = np.true_divide(process_average, float(len(process_data)))

    for idx in range(len(activation_l)):
        left = np.asarray(activation_l[idx])
        right = np.asarray(activation_r[idx])

        close_vertex = np.concatenate([left, right], axis = 0)
        generated_converted_matrix = np.dot(t_matrix, close_vertex)

        signal = np.asarray(generated_converted_matrix[channel_idx]).flatten()
        xf_original = fft(signal - np.mean(signal))
        Sxx_original = 2 * dt ** 2 / T * (xf_original * xf_original.conj())         # Compute spectrum
        Sxx_original = Sxx_original[:int(N / 2)]                                    # Ignore negative frequencies
        Sxx_original = np.expand_dims(Sxx_original, axis = 0)

        if len(generated_average) == 0:
            generated_average = Sxx_original
        else:
            generated_average = np.concatenate([generated_average, Sxx_original], axis = 0)

    fig, ax = plt.subplots(3, 1, figsize=(16, 15), sharex=True)

    ax[0].plot(faxis, raw_average.real)
    ax[0].set_title('Power Spectrum for channel {} from raw signal in epoch {}'.format(channel, epoch))
    ax[1].plot(faxis, process_average.real)
    ax[1].set_title('Power Spectrum for channel {} from filtered signal in epoch {}'.format(channel, epoch))

    for idx in range(generated_average.shape[0]):
        ax[2].plot(faxis, generated_average[idx])

    ax[2].set_title('Power Spectrum for channel {} from generated signal in epoch {}'.format(channel, epoch))

    if not eye_open:
        plt.savefig("results/img_results/{}/{}/{}/Spectrum/iteration_{:04d}/Eye_Close_{}.png".format(model_name, time, n_round, epoch, channel))    # should before show method
    else:
        plt.savefig("results/img_results/{}/{}/{}/Spectrum/iteration_{:04d}/Eye_Open_{}.png".format(model_name, time, n_round, epoch, channel))     # should before show method
    
    plt.close()

    # draw the all raw and process        
    plt.figure(figsize=(20,10))
    for signal in raw_collection:
        plt.plot(faxis, signal, linewidth = 1)
    
    plt.title("Raw signals")

    if eye_open:
        plt.savefig("results/img_results/{}/{}/{}/Spectrum/OpenRawSignals.png".format(model_name, time, n_round))
    else:
        plt.savefig("results/img_results/{}/{}/{}/Spectrum/CloseRawSignals.png".format(model_name, time, n_round))
    plt.close()

    plt.figure(figsize=(20,10))
    for signal in process_collection:
        plt.plot(faxis, signal, linewidth = 1)
    
    plt.title("Process signals")
    
    if eye_open:
        plt.savefig("results/img_results/{}/{}/{}/Spectrum/iteration_{:04d}/EyeOpen_ProcessSignals.png".format(model_name, time, n_round, epoch))
    else:
        plt.savefig("results/img_results/{}/{}/{}/Spectrum/iteration_{:04d}/EyeClose_ProcessSignals.png".format(model_name, time, n_round, epoch))
    plt.close()
    

def generate_power_spectrum(eye_open, channel, original_signal, generated_signal, epoch, time, model_name, n_round, event_idx):

    directory1 = 'results/img_results/{}/{}/{}'.format(model_name, time, n_round)
    directory2 = 'results/img_results/{}/{}/{}/Spectrum'.format(model_name, time, n_round)
    directory3 = 'results/img_results/{}/{}/{}/Spectrum/iteration_{:04d}'.format(model_name, time, n_round, epoch)
    directory4 = 'results/img_results/{}/{}/{}/Spectrum/iteration_{:04d}/Eye_Close_{:04d}'.format(model_name, time, n_round, epoch, event_idx)
    directory5 = 'results/img_results/{}/{}/{}/Spectrum/iteration_{:04d}/Eye_Open_{:04d}'.format(model_name, time, n_round, epoch, event_idx)

    if not os.path.exists(directory1):
        os.mkdir(directory1)
    
    if not os.path.exists(directory2):
        os.mkdir(directory2)

    if not os.path.exists(directory3):
        os.mkdir(directory3)

    if not os.path.exists(directory4):
        os.mkdir(directory4)

    if not os.path.exists(directory5):
        os.mkdir(directory5)

    # plot original signal
    N = original_signal.shape[0]
    dt = 0.004
    T = 2.0

    xf_original = fft(original_signal - np.mean(original_signal))
    xf_generated = fft(generated_signal - np.mean(generated_signal))
    Sxx_original = 2 * dt ** 2 / T * (xf_original * xf_original.conj())         # Compute spectrum
    Sxx_original = Sxx_original[:int(N / 2)]                                    # Ignore negative frequencies
    Sxx_generated = 2 * dt ** 2 / T * (xf_generated * xf_generated.conj())      # Compute spectrum
    Sxx_generated = Sxx_generated[:int(N / 2)]                                  # Ignore negative frequencies

    fNQ = 1 / dt / 2                            # Determine Nyquist frequency
    faxis = np.arrange(0, fNQ, 0.5)              # Construct frequency axis

    # start drawing result
    # fig, ax = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig, ax = plt.subplots(1, 1, figsize=(16, 6), sharex=True)

    # ax[0].plot(faxis, Sxx_original.real)
    # ax[0].set_title('Power Spectrum for channel {} from original signal in epoch {}'.format(channel, epoch))
    # ax[1].plot(faxis, Sxx_generated.real)
    # ax[1].set_title('Power Spectrum for channel {} from generated signal in epoch {}'.format(channel, epoch))

    ax[0].plot(faxis, Sxx_generated.real)
    ax[0].set_title('Power Spectrum for channel {} from generated signal in epoch {}'.format(channel, epoch))

    if not eye_open:
        plt.savefig("results/img_results/{}/{}/{}/Spectrum/iteration_{:04d}/Eye_Close_{:04d}/{}.png".format(model_name, time, n_round, epoch, event_idx, channel))    # should before show method
    else:
        plt.savefig("results/img_results/{}/{}/{}/Spectrum/iteration_{:04d}/Eye_Open_{:04d}/{}.png".format(model_name, time, n_round, epoch, event_idx, channel))     # should before show method
    
    plt.close()

    