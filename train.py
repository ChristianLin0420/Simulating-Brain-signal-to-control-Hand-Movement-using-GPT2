

import os
import argparse
import numpy as np
import tensorflow as tf

from datetime import datetime
from tensorflow import keras
 
from folder import check_folders
from utils.callback import EarlyStoppingAtMinLoss, RecordGeneratedImages
from utils.model_monitor import save_loss_range_record, save_result_as_gif

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
    current_round = 1

    # get datsets
    datasets = initial_mnist_datset()

    # initial model
    config = GPT2Config(n_layer = int(args.num_layer), 
                        n_head = int(args.num_head), 
                        n_embd = int(args.noise_hidden_dim), 
                        n_positions = int(args.noise_len))

    g_loss_collection = []
    d_loss_collection = []

    while round_num >= current_round:

        d_optimizer = keras.optimizers.Adam(learning_rate = 0.0003)
        g_optimizer = keras.optimizers.Adam(learning_rate = 0.0003)
        loss_fn = keras.losses.BinaryCrossentropy(from_logits = True)

        check_folders(time = time, model_name = args.mode)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = './logs/' + str(args.mode) + '/' + str(time) + '/' + str(current_round), histogram_freq = 1)

        file_writer = tf.summary.create_file_writer('./logs/' + str(args.mode) + '/' + str(time) + "/" + str(current_round) + "/metrics")
        file_writer.set_as_default()

        # create git to observe the training performance
        save_result_as_gif(time, args.mode, current_round)

        if args.mode == "gpt2gan": 

            model = gpt2gan(
                config = config,
                noise_len = int(args.noise_len),
                noise_dim = int(args.noise_hidden_dim)
            )
            print(model.config)

            model.build(datasets.shape)

            model.compile(
                d_optimizer = d_optimizer,
                g_optimizer = g_optimizer,
                loss_fn = loss_fn
            )

            history = model.fit(
                datasets.take(80), 
                epochs = int(args.epochs), 
                verbose = 1, 
                callbacks = [EarlyStoppingAtMinLoss(), RecordGeneratedImages(time, current_round, args.mode), tensorboard_callback]
            )

            g_loss = history.history['g_loss']
            d_loss = history.history['d_loss']

            g_loss_collection.append(g_loss)
            d_loss_collection.append(d_loss)
            
            # save model
            model.save("./trained_model/" + str(args.mode) + "/" + str(time) + "/model_" + str(current_round))

        elif args.mode == "gpt2wgan":

            model = gpt2wgan(
                config = config,
                noise_len = int(args.noise_len),
                noise_dim = int(args.noise_hidden_dim)
            )
            print(model.config)

            model.build(datasets.shape)

            model.compile(
                d_optimizer = d_optimizer,
                g_optimizer = g_optimizer,
                loss_fn = loss_fn
            )

            history = model.fit(
                datasets.take(80), 
                epochs = int(args.epochs), 
                verbose = 1, 
                callbacks = [EarlyStoppingAtMinLoss(), RecordGeneratedImages(time, current_round, args.mode), tensorboard_callback]
            )

            g_loss = history.history['g_loss']
            d_loss = history.history['d_loss']

            g_loss_collection.append(g_loss)
            d_loss_collection.append(d_loss)

            # save model
            tf.saved_model.save(model, "./trained_model/" + str(args.mode) + "/" + str(time) + "/model_" + str(current_round))
            # model.save("./trained_model/" + str(args.mode) + "/" + str(time) + "/model_" + str(current_round))

        else:
            print("[Error] Should specify one training mode!!!")
            break
        
        # create git to observe the training performance
        tf.saved_model.save(model, "./trained_model/" + str(args.mode) + "/" + str(time) + "/model_" + str(current_round))
        # save_result_as_gif(time, args.mode, current_round)
        
        current_round += 1


    # save figure
    if args.mode == "gpt2gan":
        save_loss_range_record(np.arange(len(g_loss_collection[0])), g_loss_collection, time, args.mode, "g_loss")
        save_loss_range_record(np.arange(len(d_loss_collection[0])), d_loss_collection, time, args.mode, "d_loss")
    elif args.mode == "gpt2wgan":
        save_loss_range_record(np.arange(len(g_loss_collection[0])), g_loss_collection, time, args.mode, "g_loss")
        save_loss_range_record(np.arange(len(d_loss_collection[0])), d_loss_collection, time, args.mode, "d_loss")