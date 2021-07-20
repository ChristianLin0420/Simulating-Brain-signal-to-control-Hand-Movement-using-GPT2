


import os
import glob
import imageio
import matplotlib.pyplot as plt

def generate_and_save_images(model, time, epoch, test_input, model_name):

    diractory = 'results/gpt2_results/{}_{}'.format(time)

    if not os.path.exists(diractory):
        os.mkdir(diractory)

    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training = False)

    _ = plt.figure(figsize = (4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap = 'gray')
        plt.axis('off')

    plt.savefig('results/gpt2_results/{}_{}/image_at_epoch_{:04d}.png'.format(model_name, time, epoch))
    plt.close()


def save_result_as_gif(time, model_name):

    diractory = 'results/gpt2_results/{}_{}'.format(model_name, time)

    if not os.path.exists(diractory):
        os.mkdir(diractory)

    anim_file = 'results/gpt2_results/{}_{}/gpt2_dcgan.gif'.format(model_name, time)

    with imageio.get_writer(anim_file, mode = 'I') as writer:
        filenames = glob.glob('results/gpt2_results/{}_{}/image*.png'.format(model_name, time))
        filenames = sorted(filenames)

        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

def save_loss_record(lst_iter, g_loss, d_loss, time, model_name):

    diractory = 'results/gpt2_loss/{}_{}'.format(model_name, time)

    if not os.path.exists(diractory):
        os.mkdir(diractory)

    title = "gpt2_gan_loss_record"

    plt.plot(lst_iter, g_loss, '-b', label='loss')
    plt.plot(lst_iter, d_loss, '-r', label='accuracy')

    plt.xlabel("n iteration")
    plt.legend(loc = 'upper left')
    plt.title(title)

    # save image
    plt.savefig("results/gpt2_loss/{}_{}.png".format(model_name, time))  # should before show method
    plt.close()