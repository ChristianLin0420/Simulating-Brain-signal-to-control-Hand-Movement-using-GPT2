


import os
import glob
import imageio
import numpy as np
import matplotlib.pyplot as plt

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

def save_loss_range_record(lst_iter, loss, time, model_name, line_name):
    directory = 'results/training_loss/{}/{}'.format(model_name, time)

    if not os.path.exists(directory):
        os.mkdir(directory)

    title = "{}_average".format(line_name)

    l = np.asarray(loss)
    mean = np.mean(l, axis = 0)
    standard_dev = np.std(l, axis = 0)

    plt.figure(figsize=(15,5))
    plt.plot(lst_iter, mean, '-')
    plt.fill_between(lst_iter, mean - standard_dev, mean + standard_dev, alpha = 0.2)

    plt.xlabel("n iteration")
    plt.legend(loc = 'upper left')
    plt.title(title)

    # save image
    plt.savefig("results/training_loss/{}/{}/{}.png".format(model_name, time, line_name))  # should before show method
    plt.close()

    
