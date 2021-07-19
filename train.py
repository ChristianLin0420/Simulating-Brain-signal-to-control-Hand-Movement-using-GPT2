

import enum
import os
import glob
import imageio
import argparse

import time as tt
import tensorflow as tf
import matplotlib.pyplot as plt

from typing import Any
from datetime import datetime
from alive_progress import alive_bar

from model.GPT2GAN import GPT2GAN
from config.config_gpt2 import GPT2Config


def generate_and_save_images(model, time, epoch, test_input):

    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training = False)

    _ = plt.figure(figsize = (4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap = 'gray')
        plt.axis('off')

    plt.savefig('results/gpt2_result/{}/image_at_epoch_{:04d}.png'.format(time, epoch))
    plt.show()

def save_loss_record(lst_iter, g_loss, d_loss, time):

    title = "gpt2_gan_loss_record"

    plt.plot(lst_iter, g_loss, '-b', label='loss')
    plt.plot(lst_iter, d_loss, '-r', label='accuracy')

    plt.xlabel("n iteration")
    plt.legend(loc = 'upper left')
    plt.title(title)

    # save image
    path = ""
    plt.savefig("results/gpt2_loss/{}.png".format(time))  # should before show method

    # show
    plt.show()

def initial_mnist_datset(buffer_size: int = 1000, batch_size: int = 8):
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size)
    return train_dataset

def discriminator_loss(real_output, fake_output, cross_entropy):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output, cross_entropy):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step( images, 
                batch_size: int = 8, 
                noise_len: int = 784, 
                noise_dim: int = 32, 
                cross_entropy: Any = None,
                generator_optimizer: Any = None, 
                discriminator_optimizer: Any = None):

    noise = tf.random.normal([batch_size, noise_len, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = model.generator(noise, training = True)

        real_output = model.discriminator(images, training = True)
        fake_output = model.discriminator(generated_images, training = True)

        gen_loss = generator_loss(fake_output, cross_entropy)
        disc_loss = discriminator_loss(real_output, fake_output, cross_entropy)

    gradients_of_generator = gen_tape.gradient(gen_loss, model.generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, model.discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, model.generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, model.discriminator.trainable_variables))

    return float(gen_loss), float(disc_loss)

def train(model, dataset, time, epochs, batch_size, num_examples_to_generate, noise_len, noise_dim):

    seed = tf.random.normal([num_examples_to_generate, noise_len, noise_dim])

    # This method returns a helper function to compute cross entropy loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)

    # setup optimizers
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint_dir = './checkpoints/gpt2_training_checkpoints/{}'.format(time)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                                     discriminator_optimizer = discriminator_optimizer,
                                     generator = model.generator,
                                     discriminator = model.discriminator)

    gen_loss_record = []
    dis_loss_record = []

    with alive_bar(epochs) as bar:
        for i in range(epochs):
            print("epoch: {}".format(i))

            start = tt.time()
            
            for image_batch in dataset:
                g_l, d_l = train_step(  images = image_batch, 
                                        batch_size = batch_size, 
                                        noise_len = noise_len, 
                                        noise_dim = noise_dim, 
                                        cross_entropy = cross_entropy, 
                                        generator_optimizer = generator_optimizer, 
                                        discriminator_optimizer = discriminator_optimizer)

                gen_loss_record.append(g_l)
                dis_loss_record.append(d_l)

            # Produce images for the GIF as you go
            # display.clear_output(wait=True)
            generate_and_save_images(model.generator, time, i + 1, seed)

            # Save the model every 15 epochs
            if (i + 1) % 10 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)

            print ('Time for epoch {} is {} sec'.format(i + 1, tt.time() - start))
        
            bar()

        # Generate after the final epoch
        # display.clear_output(wait=True)
        print("epoch: {}".format(i))
        generate_and_save_images(model.generator, time, epochs, seed)
        print("finished generating images")
        print("-" * 80)

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    assert len(gen_loss_record) == len(dis_loss_record)

    save_loss_record(len(gen_loss_record), gen_loss_record, dis_loss_record, time)
    save_result_as_gif(time)

def save_result_as_gif(time):
    anim_file = 'results/gpt2_result/{}/gpt2_dcgan.gif'.format(time)

    with imageio.get_writer(anim_file, mode = 'I') as writer:
        filenames = glob.glob('results/gpt2_result/{}/image*.png')
        filenames = sorted(filenames)

        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer_size", default = 1000)
    parser.add_argument("--batch_size", default = 8)
    parser.add_argument("--epochs", default = 50)
    parser.add_argument("--noise_len", default = 784)
    parser.add_argument("--noise_hidden_dim", default = 32)
    parser.add_argument("--example_to_generate", default = 16)
    parser.add_argument("--use_gpu", default = True)  
    parser.add_argument("--load_model", default = False)              
    args = parser.parse_args()

    width = 20
    print("{0: <{width}}: {val}".format("buffer_size", width = width, val = args.buffer_size))
    print("{0: <{width}}: {val}".format("batch_size", width = width, val = args.batch_size))
    print("{0: <{width}}: {val}".format("epochs", width = width, val = args.epochs))
    print("{0: <{width}}: {val}".format("noise_len", width = width, val = args.noise_len))
    print("{0: <{width}}: {val}".format("noise_hidden_dim", width = width, val = args.noise_hidden_dim))
    print("{0: <{width}}: {val}".format("example_to_generate", width = width, val = args.example_to_generate))
    print("{0: <{width}}: {val}".format("use_gpu", width = width, val = args.use_gpu))
    print("{0: <{width}}: {val}".format("load_model", width = width, val = args.load_model))

    # if use_gpu is TRUE, then get one and run the program on it
    if args.use_gpu:
        try:
            gpus = tf.config.list_physical_devices(device_type = 'GPU')
            
            if gpus:
                tf.config.set_visible_devices(devices = gpus[0], device_type = 'GPU')
        except:
            print("[No GPR] there is no availible gpu to use!!!")

    # get datsets
    datsets = initial_mnist_datset()

    # initial model
    config = GPT2Config(n_head = 4, n_embd = args.noise_hidden_dim, n_positions = args.noise_len)
    model = GPT2GAN(config = config)

    if args.load_model:
        pass
    
    print(model.config)

    # training step
    now = datetime.now()
    time = now.strftime("%d_%m_%Y_%H_%M_%S")

    print("current time is: {}".format(time))

    train(  model = model, 
            dataset = datsets, 
            time = time, 
            epochs = args.epochs, 
            batch_size = args.batch_size,
            num_examples_to_generate = args.example_to_generate, 
            noise_len = args.noise_len, 
            noise_dim = args.noise_hidden_dim  )

