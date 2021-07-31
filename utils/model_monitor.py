


import os
import glob
import json
from posixpath import join
import imageio
import numpy as np
import matplotlib.pyplot as plt

from logging import error

from config.config_gpt2 import GPT2Config

def show_generated_image(prediction):
    
    _ = plt.figure()

    plt.imshow(prediction[:, :, :, 0] * 127.5 + 127.5, cmap = 'gray')
    plt.axis('off')

    plt.show()

def save_random_vector(path, seed, name):

    if not os.path.exists(path):
        os.mkdir(path)

    filepath = path + "/" + str(name) + ".txt"

    tmp = np.asarray(seed)
    data = {"data": tmp.reshape(-1)}
    
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
        tmp = np.asarray(arr)
        vector = tmp.reshape([1, noise_len, noise_dim])

        return vector

    return None

def generate_and_save_images(predictions, time, n_round, epoch, model_name):

    diractory = 'results/img_results/{}/{}/{}'.format(model_name, time, n_round)

    if not os.path.exists(diractory):
        os.mkdir(diractory)

    _ = plt.figure(figsize = (4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap = 'gray')
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

    plt.figure(figsize=(15,5))
    plt.plot(lst_iter, g_loss, '-b', label = 'generator_loss', linewidth = 2)
    plt.plot(lst_iter, d_loss, '-r', label = 'discriminator_loss', linewidth = 2)

    plt.xlabel("n iteration")
    plt.legend(loc = 'upper left')
    plt.title(title)

    # save image
    plt.savefig("results/training_loss/{}/{}/{}/loss.png".format(model_name, time, n_round))  # should before show method
    plt.close()

