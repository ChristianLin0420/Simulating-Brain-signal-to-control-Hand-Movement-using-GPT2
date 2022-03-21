


import os
import glob
import json
from typing import Dict
import imageio
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from logging import error

from .brain_activation import restore_brain_activation, generate_eeg 

def show_avg_history(title: str, records: Dict, path: str):
    pass

def show_generated_image(prediction):
    
    _ = plt.figure()

    plt.imshow(prediction[0] * 127.5 + 127.5, cmap = 'gray')
    plt.axis('off')

    plt.show()

def save_random_vector(path, seed, name):

    if not os.path.exists(path):
        os.mkdir(path)

    filepath = path + "/" + str(name) + ".txt"

    # tmp = np.asarray(seed)
    data = {"data": seed}
    
    jsonStr = json.dumps(data)

    print("data: {}".format(data))

    with open(filepath, 'w') as f:
        f.write(jsonStr)

def load_random_vector(path, noise_len: int = 784, noise_dim: int = 32):
    
    if not os.path.exists(path):
        error("Given file path is not existed")
    else:
        with open(path) as f:
            data = json.load(f)
        
        arr = data["data"]
        # tmp = np.asarray(arr)
        # vector = tmp.reshape([1, noise_len, noise_dim])

        return arr

    return None

def generate_and_save_images(predictions, time, n_round, epoch, model_name):

    diractory = 'results/img_results/{}/{}/{}'.format(model_name, time, n_round)

    if not os.path.exists(diractory):
        os.mkdir(diractory)

    _ = plt.figure(figsize = (4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        
        if predictions.shape[-1] == 1:
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap = 'gray')
        elif predictions.shape[-1] == 3:
            pred = np.asarray(predictions[i, :, :, :])
            predictions[i, :, :, :] = (predictions[i, :, :, :] * 127.5 + 127.5) / 255.0
            plt.imshow(predictions[i, :, :, :])
        else:
            error("Last dimension of the prediction is invalid")
            
        plt.axis('off')

    plt.savefig('results/img_results/{}/{}/{}/image_at_epoch_{:04d}.png'.format(model_name, time, n_round, epoch))
    plt.close()

    


def save_result_as_gif(time, model_name, n_round):

    diractory = 'results/img_results/{}/{}/{}'.format(model_name, time, n_round)

    if not os.path.exists(diractory):
        os.mkdir(diractory)

    anim_file = 'results/img_results/{}/{}/{}/gpt2_dcgan.gif'.format(model_name, time, n_round)

    with imageio.get_writer(anim_file, mode = 'I') as writer:
        filenames = glob.glob('results/img_results/{}/{}/{}/image*.png'.format(model_name, time, n_round))
        filenames = sorted(filenames)

        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

def save_loss_record(lst_iter, g_loss, d_loss, time, model_name, n_round):

    diractory = 'results/training_loss/{}/{}/{}'.format(model_name, time, n_round)

    if not os.path.exists(diractory):
        os.mkdir(diractory)

    title = "{}_loss".format(model_name)

    plt.figure(figsize=(8,5))
    plt.plot(lst_iter, g_loss, '-b', label = 'generator_loss', linewidth = 1)
    plt.plot(lst_iter, d_loss, '-r', label = 'discriminator_loss', linewidth = 1)

    plt.xlabel("n iteration")
    plt.legend(loc = 'upper left')
    plt.title(title)

    # save image
    plt.savefig("results/training_loss/{}/{}/{}/loss.png".format(model_name, time, n_round))  # should before show method
    plt.close()


def save_loss_range_record(lst_iter, loss_1, loss_2, time, model_name, line_name, line_name2):
    directory = 'results/training_loss/{}/{}'.format(model_name, time)

    if not os.path.exists(directory):
        os.mkdir(directory)

    title = "{}_average".format(line_name)

    l = np.asarray(loss_1)

    if loss_2 is not None:
        ll = np.asarray(loss_2)
    # print("l shape: {}".format(l.shape))

    mean_1 = np.mean(l, axis = 0)
    standard_dev_1 = np.std(l, axis = 0)

    if loss_2 is not None:
        mean_2 = np.mean(ll, axis = 0)
        standard_dev_2 = np.std(ll, axis = 0)

    plt.figure(figsize=(8,5))
    plt.plot(lst_iter, mean_1, '-b', label = line_name, linewidth = 1)
    plt.fill_between(lst_iter, mean_1 - standard_dev_1, mean_1 + standard_dev_1, color = 'blue', alpha = 0.2)
    
    if loss_2 is not None:
        plt.plot(lst_iter, mean_2, '-r', label = line_name2, linewidth = 1)
        plt.fill_between(lst_iter, mean_2 - standard_dev_2, mean_2 + standard_dev_2, color = 'red', alpha = 0.2)

    plt.xlabel("n iteration")
    plt.legend(loc = 'upper left')
    plt.title(title)

    # save image
    plt.savefig("results/training_loss/{}/{}/{}.png".format(model_name, time, line_name))  # should before show method
    plt.close()

def save_distribution_record(data, epoch, time, model_name, n_round):

    diractory = 'results/img_results/{}/{}/{}'.format(model_name, time, n_round)

    real_close_data = data[0]
    real_open_data = data[1]
    generated_close_data = data[2]
    generated_open_data = data[3]

    real_close_data = np.asarray(real_close_data).flatten()
    real_open_data = np.asarray(real_open_data).flatten()
    generated_close_data = np.asarray(generated_close_data).flatten()
    generated_open_data = np.asarray(generated_close_data).flatten()

    if not os.path.exists(diractory):
        os.mkdir(diractory)

    directory = 'results/img_results/{}/{}/{}/distribuion'.format(model_name, time, n_round)
    directory1 = 'results/img_results/{}/{}/{}/distribuion/image_at_epoch_{:04d}'.format(model_name, time, n_round, epoch)

    if not os.path.exists(directory):
        os.mkdir(directory)

    if not os.path.exists(directory1):
        os.mkdir(directory1)

    title = "{}-Eye Close Distribution".format(model_name)

    plt.figure(figsize = (8, 5))
    sns.kdeplot(real_close_data)
    sns.kdeplot(generated_close_data)

    plt.xlabel("generated values")
    plt.title(title)
    plt.ylim(0, 40)
    plt.xlim(0, 1.0)

    plt.savefig('results/img_results/{}/{}/{}/distribuion/image_at_epoch_{:04d}/Eye_Close.png'.format(model_name, time, n_round, epoch))
    plt.close()

    title = "{}-Eye Open Distribution".format(model_name)

    plt.figure(figsize = (8, 5))
    sns.kdeplot(real_open_data)
    sns.kdeplot(generated_open_data)

    plt.xlabel("generated values")
    plt.title(title)
    plt.ylim(0, 40)
    plt.xlim(0, 1.0)

    plt.savefig('results/img_results/{}/{}/{}/distribuion/image_at_epoch_{:04d}/Eye_Open.png'.format(model_name, time, n_round, epoch))
    plt.close()

    real_close_data = None
    real_open_data = None
    generated_close_data = None
    generated_open_data = None

    del real_close_data
    del real_open_data
    del generated_close_data
    del generated_open_data


def record_model_weight(weights):

    directory = 'weight'

    if not os.path.exists(directory):
        os.mkdir(directory)

    step = 0

    while os.path.exists('weight/{}.txt'.format(step)):
        step += 1

    if step == 100:
        return

    f = open("weight/{}.txt".format(step), "x")

    # print("weights length: {}".format(len(weights)))

    # count = 0

    # content = ""

    print(type(weights))
    f.write(str(weights))

    # for item in weights:
    #     count += 1
    #     f.write(str(item))
        # print("item length: {}".format(len(item)))

    # print("total weight items is {}".format(count))
    f.close()

    # print(weights)


def save_spectrum(X, epoch, time, model_name, n_round):

    Zxx = tf.signal.stft(X, frame_length=256, frame_step=16)
    Zxx = tf.abs(Zxx)

    samples = 0
    log_spec = tf.math.log(tf.transpose(Zxx[samples][0]))
    height = 40
    width = log_spec.shape[1]
    x_axis = tf.linspace(0, 2, num=width)
    y_axis = range(height)
    plt.pcolormesh(x_axis, y_axis, log_spec[:40, ])
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    
    plt.close()


def save_ground_truth():
    pass

    
