

import os
import argparse
import tensorflow as tf

from datetime import datetime
from tensorflow import keras
 
from folder import check_folders
from utils.callback import EarlyStoppingAtMinLoss

from model.gpt2gan import gpt2gan
from model.gpt2wgan import gpt2wgan
from config.config_gpt2 import GPT2Config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def initial_mnist_datset(buffer_size: int = 1000, batch_size: int = 8):
    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size = buffer_size).batch(batch_size)
    return train_dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default = "gpt2gan")
    parser.add_argument("--buffer_size", default = 1000)
    parser.add_argument("--batch_size", default = 8)
    parser.add_argument("--epochs", default = 200)
    parser.add_argument("--noise_len", default = 784)
    parser.add_argument("--noise_hidden_dim", default = 32)
    parser.add_argument("--example_to_generate", default = 16)
    parser.add_argument("--num_layer", default = 2)
    parser.add_argument("--num_head", default = 4)
    parser.add_argument("--num_round", default = 3)
    parser.add_argument("--use_gpu", default = True)  
    parser.add_argument("--load_model", default = False)              
    args = parser.parse_args()

    width = 20
    print("{0: <{width}}: {val}".format("mode", width = width, val = args.mode))
    print("{0: <{width}}: {val}".format("buffer_size", width = width, val = args.buffer_size))
    print("{0: <{width}}: {val}".format("batch_size", width = width, val = args.batch_size))
    print("{0: <{width}}: {val}".format("epochs", width = width, val = args.epochs))
    print("{0: <{width}}: {val}".format("noise_len", width = width, val = args.noise_len))
    print("{0: <{width}}: {val}".format("noise_hidden_dim", width = width, val = args.noise_hidden_dim))
    print("{0: <{width}}: {val}".format("example_to_generate", width = width, val = args.example_to_generate))
    print("{0: <{width}}: {val}".format("num_layer", width = width, val = args.num_layer))
    print("{0: <{width}}: {val}".format("num_head", width = width, val = args.num_head))
    print("{0: <{width}}: {val}".format("num_round", width = width, val = args.num_round))
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

    # get environment parameters
    now = datetime.now()
    time = now.strftime("%d_%m_%Y_%H_%M_%S")
    round_num = int(args.num_round)

    # get datsets
    datasets = initial_mnist_datset()

    # initial model
    config = GPT2Config(n_layer = int(args.num_layer), 
                        n_head = int(args.num_head), 
                        n_embd = int(args.noise_hidden_dim), 
                        n_positions = int(args.noise_len))


    while round_num > 0:

        d_optimizer = keras.optimizers.Adam(learning_rate = 0.0003)
        g_optimizer = keras.optimizers.Adam(learning_rate = 0.0003)
        loss_fn = keras.losses.BinaryCrossentropy(from_logits = True)

        if args.mode == "gpt2gan": 

            check_folders(time = time, model_name = 'gan')
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = './logs/gpt2gan/' + str(time), histogram_freq = 1)

            file_writer = tf.summary.create_file_writer('./logs/gpt2gan/' + str(time) + "/metrics")
            file_writer.set_as_default()

            model = gpt2gan(
                config = config,
                noise_len = int(args.noise_len),
                noise_dim = int(args.noise_hidden_dim)
            )
            print(model.config)

            model.compile(
                d_optimizer = d_optimizer,
                g_optimizer = g_optimizer,
                loss_fn = loss_fn
            )

            history = model.fit(
                datasets.take(80), 
                epochs = int(args.epochs), 
                verbose = 1, 
                callbacks = [EarlyStoppingAtMinLoss(), tensorboard_callback]
            )

            g_loss = history.history['g_loss']
            d_loss = history.history['d_loss']

        elif args.mode == "gpt2wgan":
            
            check_folders(time = time, model_name = 'wgan')

            pass

        else:
            print("[Error] Should specify one training mode!!!")
            break

        round_num -= 1
