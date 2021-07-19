
import os
import tensorflow as tf

import time
import glob
import PIL
import imageio
import matplotlib.pyplot as plt

from datetime import datetime
from IPython import display
from alive_progress import alive_bar


from model.GPT2GAN import GPT2GAN
from config.config_gpt2 import GPT2Config



print(tf.config.list_physical_devices())

gpus = tf.config.list_physical_devices(device_type='GPU')

if gpus:
    tf.config.set_visible_devices(devices=gpus[0], device_type='GPU')
else:
    print("[No GPR] there is no availible gpu to use!!!")

BUFFER_SIZE = 1000
BATCH_SIZE = 8
EPOCHS = 50
noise_len = 784
noise_dim = 32
num_examples_to_generate = 16

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y/%H:%M:%S")
print("date and time =", dt_string)


(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

config = GPT2Config(n_head = 4, n_embd = noise_dim, n_positions=noise_len)
model = GPT2GAN(config = config)

print(model.config)

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_len, noise_dim])

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './gpt2_training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                                 discriminator_optimizer = discriminator_optimizer,
                                 generator = model.generator,
                                 discriminator = model.discriminator)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_len, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = model.generator(noise, training=True)

        real_output = model.discriminator(images, training=True)
        fake_output = model.discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, model.generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, model.discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, model.generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, model.discriminator.trainable_variables))


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('gpt2_result/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

def train(dataset, epochs):
    with alive_bar(epochs) as bar:
        for i in range(epochs):
            print("epoch: {}".format(i))

            start = time.time()
            
            for image_batch in dataset:
                train_step(image_batch)

            # Produce images for the GIF as you go
            # display.clear_output(wait=True)
            generate_and_save_images(model.generator, i + 1, seed)

            # Save the model every 15 epochs
            if (i + 1) % 10 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)

            print ('Time for epoch {} is {} sec'.format(i + 1, time.time() - start))
        
            bar()

        # Generate after the final epoch
        # display.clear_output(wait=True)
        print("epoch: {}".format(i))
        generate_and_save_images(model.generator, epochs, seed)
        print("finished generating images")
        print("-" * 80)

# Display a single image using the epoch number
def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

train(train_dataset, EPOCHS)

# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# display_image(EPOCHS)

anim_file = 'gpt2_result/gpt2_dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('gpt2_result/image*.png')
    filenames = sorted(filenames)

    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        

