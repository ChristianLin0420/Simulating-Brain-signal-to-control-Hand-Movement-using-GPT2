

from logging import error
import os
import time as tt
import argparse
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from datetime import datetime
from tensorflow import keras
 
from folder import check_folders
from utils.callback import EarlyStoppingAtMinLoss, RecordGeneratedImages
from utils.model_monitor import save_loss_range_record, save_loss_record, save_random_vector, save_result_as_gif, show_generated_image

from model.gpt2gan import gpt2gan
from model.gpt2wgan import gpt2wgan
from model.gpt2cgan import gpt2cgan
from model.model_utils import load_model
from config.config_gpt2 import GPT2Config, save_model_config


LEARNING_RATE = 0.0003


def initial_mnist_datset(buffer_size: int = 1000, batch_size: int = 8):
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    # train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    train_images = np.repeat(train_images, 3, axis = 3)
    train_labels = keras.utils.to_categorical(train_labels, 10)

    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(buffer_size = buffer_size).batch(batch_size)
    return train_dataset, np.shape(np.asarray(train_dataset))

def dataset_np(last_dim: int = 1):
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    if last_dim == 3:
        train_images = np.repeat(train_images, 3, axis = 3)

    train_labels = keras.utils.to_categorical(train_labels, 10)

    return (train_images, train_labels)

def load_dataset():
    start_time = tt.time()

    ROOT_DIR = '/home/jupyter-ivanljh123/rsc/Source Estimate'

    _, dirs, files = os.walk(ROOT_DIR).__next__()

    train_data = np.asarray([])
    train_label = np.asarray([])

    dirs = dirs[:40]

    for dir in dirs:
        path = ROOT_DIR + '/' + dir
        _, dirs, files = os.walk(path).__next__()

        if len(files) == 2:
            eye_open_filename = files[0]
            eye_close_filename = files[1]

            eye_open_path = path + '/' + eye_open_filename
            eye_close_path = path + '/' + eye_close_filename

            eye_open_data = np.load(eye_open_path, allow_pickle = True)
            eye_close_data = np.load(eye_close_path, allow_pickle = True)

            left_open_epoch = len(eye_open_data["left"])
            right_open_epoch = len(eye_open_data["right"])
            left_close_epoch = len(eye_close_data["left"])
            right_close_epoch = len(eye_close_data["right"])

            open_epoch = min(left_open_epoch, right_open_epoch)
            close_epoch = min(left_close_epoch, right_close_epoch)
            vertex_count = 1022
            timestemp = 500

            left_open_data = np.asarray(eye_open_data["left"][:open_epoch, :vertex_count, :timestemp])
            right_open_data = np.asarray(eye_open_data["right"][:open_epoch, :vertex_count, :timestemp])
            left_close_data = np.asarray(eye_close_data["left"][:close_epoch, :vertex_count, :timestemp])
            right_close_data = np.asarray(eye_close_data["right"][:close_epoch, :vertex_count, :timestemp])

            eye_open_data = np.concatenate((left_open_data, right_open_data), axis = 1)
            eye_close_data = np.concatenate((left_close_data, right_close_data), axis = 1)

            if len(train_data) == 0 and len(train_label) == 0:
                train_data = eye_open_data
                train_label = np.asarray([1] * open_epoch) 
                train_data = np.concatenate((train_data, eye_close_data), axis = 0)
                train_label = np.concatenate((train_label, np.asarray([0] * close_epoch)), axis = 0)
            else:
                assert train_data.shape[0] == train_label.shape[0]

                train_data = np.concatenate((train_data, eye_open_data), axis = 0)
                train_label = np.concatenate((train_label, np.asarray([1] * open_epoch)), axis = 0)
                train_data = np.concatenate((train_data, eye_close_data), axis = 0)
                train_label = np.concatenate((train_label, np.asarray([0] * close_epoch)), axis = 0)
    
    assert train_data.shape[0] == train_label.shape[0]

    p = np.random.permutation(train_data.shape[0])
    train_data = train_data[p]
    train_label = train_label[p]

    train_data = np.reshape(train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2], 1])
    train_label = keras.utils.to_categorical(train_label, 2)

    print("-" * 100)
    print(train_data.shape)
    print(train_label.shape)

    data_count = int(train_data.shape[0] / 8)
    data_count = data_count * 8

    print("--- %s seconds ---" % (tt.time() - start_time))

    return (train_data[:data_count], train_label[:data_count])

def training(args, datasets, time, num_classes: int = 2):
    
    # initial model
    config = GPT2Config(n_layer = int(args.num_layer), 
                        n_head = int(args.num_head), 
                        n_embd = int(args.noise_hidden_dim), 
                        n_positions = int(args.noise_len))

    g_loss_collection = []
    d_loss_collection = []

    batch_size = int(args.batch_size)
    round_num = int(args.num_round)
    current_round = 1
    last_dim = int(args.num_last_dim)
    add_class_dim = False

    while round_num >= current_round:

        d_optimizer = keras.optimizers.Adam(learning_rate = LEARNING_RATE)
        g_optimizer = keras.optimizers.Adam(learning_rate = LEARNING_RATE)
        loss_fn = keras.losses.BinaryCrossentropy(from_logits = True)

        check_folders(time = time, model_name = args.model)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = './logs/' + str(args.model) + '/' + str(time) + '/' + str(current_round), histogram_freq = 1)

        file_writer = tf.summary.create_file_writer('./logs/' + str(args.model) + '/' + str(time) + "/" + str(current_round) + "/metrics")
        file_writer.set_as_default()

        if args.model == "gpt2gan": 

            model = gpt2gan(
                config = config,
                noise_len = int(args.noise_len),
                noise_dim = int(args.noise_hidden_dim)
            )

            print(model.config)

            # model.build(shape)

            model.compile(
                d_optimizer = d_optimizer,
                g_optimizer = g_optimizer,
                loss_fn = loss_fn
            )

            history = model.fit(
                datasets, 
                epochs = int(args.epochs), 
                verbose = 1, 
                callbacks = [EarlyStoppingAtMinLoss(), RecordGeneratedImages(time, current_round, args.model), tensorboard_callback]
            )

            g_loss = history.history['g_loss']
            d_loss = history.history['d_loss']

            g_loss_collection.append(g_loss)
            d_loss_collection.append(d_loss)
            
            # save training loss figure
            save_loss_record(np.arange(1, len(g_loss) + 1), g_loss, d_loss, time, str(args.model), current_round)
            
            # save model
            save_model_config(config, str(args.model), time, current_round)
            model.save_weights("./trained_model/" + str(args.model) + "/" + str(time) + "/model_" + str(current_round), save_format = 'tf')

        elif args.model == "gpt2wgan":

            model = gpt2wgan(
                config = config,
                noise_len = int(args.noise_len),
                noise_dim = int(args.noise_hidden_dim)
            )
            print(model.config)

            # model.build(shape)

            model.compile(
                d_optimizer = d_optimizer,
                g_optimizer = g_optimizer,
                loss_fn = loss_fn
            )

            history = model.fit(
                datasets, 
                epochs = int(args.epochs), 
                verbose = 1, 
                callbacks = [EarlyStoppingAtMinLoss(), RecordGeneratedImages(time, current_round, args.model), tensorboard_callback]
            )

            g_loss = history.history['g_loss']
            d_loss = history.history['d_loss']

            g_loss_collection.append(g_loss)
            d_loss_collection.append(d_loss)

            # save training loss figure
            save_loss_record(np.arange(1, len(g_loss) + 1), g_loss, d_loss, time, str(args.model), current_round)

            # save model
            save_model_config(config, str(args.model), time, current_round)
            model.save_weights("./trained_model/" + str(args.model) + "/" + str(time) + "/model_" + str(current_round), save_format = 'tf')

        elif args.model == "gpt2cgan":
            
            if not add_class_dim:
                config.n_embd += num_classes
                add_class_dim = True

            model = gpt2cgan(
                config = config,
                noise_len = int(args.noise_len),
                noise_dim = int(args.noise_hidden_dim), 
                last_dim = last_dim
            )

            print(model.config)

            model.compile(
                d_optimizer = d_optimizer,
                g_optimizer = g_optimizer,
                loss_fn = loss_fn
            )

            history = model.fit(
                x = datasets[0],
                y = datasets[1],
                batch_size = batch_size,
                epochs = int(args.epochs), 
                verbose = 1, 
                callbacks = [EarlyStoppingAtMinLoss()]#, RecordGeneratedImages(time, current_round, args.model)]#, tensorboard_callback]
            )

            g_loss = history.history['g_loss']
            d_loss = history.history['d_loss']

            g_loss_collection.append(g_loss)
            d_loss_collection.append(d_loss)

            # save training loss figure
            save_loss_record(np.arange(1, len(g_loss) + 1), g_loss, d_loss, time, str(args.model), current_round)

            # save model
            save_model_config(config, str(args.model), time, current_round)
            model.save_weights("./trained_model/" + str(args.model) + "/" + str(time) + "/model_" + str(current_round), save_format = 'tf')

        else:
            print("[Error] Should specify one training model!!!")
            break
        
        # create git to observe the training performance
        save_result_as_gif(time, args.model, current_round)
        
        current_round += 1


    # save average training loss figure
    if args.model == "gpt2gan":
        save_loss_range_record(np.arange(len(g_loss_collection[0])), g_loss_collection, time, args.model, "g_loss")
        save_loss_range_record(np.arange(len(d_loss_collection[0])), d_loss_collection, time, args.model, "d_loss")
    elif args.model == "gpt2wgan":
        save_loss_range_record(np.arange(len(g_loss_collection[0])), g_loss_collection, time, args.model, "g_loss")
        save_loss_range_record(np.arange(len(d_loss_collection[0])), d_loss_collection, time, args.model, "d_loss")
    elif args.model == "gpt2cgan":
        save_loss_range_record(np.arange(len(g_loss_collection[0])), g_loss_collection, time, args.model, "g_loss")
        save_loss_range_record(np.arange(len(d_loss_collection[0])), d_loss_collection, time, args.model, "d_loss")
        

def find_random_vector(model_path, model, noise_len: int = 784, noise_dim: int = 32):

    model = load_model(model_path, model, noise_len, noise_dim)

    if model is None:
        error("Return model is None")
    else:
        end = False

        while not end:
            seed = np.random.random([1, noise_len, noise_dim])
            
            prediction = model.generator(seed, training = False)
            print(prediction.shape)
            print(prediction)
            show_generated_image(prediction)

            available = input("is this generated image avaliable(no/yes): ")

            if available == 'yes':
                vector_name = input("please enter image name: ")
                print("vector_name: {}".format(vector_name))
                save_random_vector(model_path, seed.tolist(), vector_name)
            elif available == "":
                end = True
            else:
                print("Restart generating image by random vector")
            
            plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default = 'training')
    parser.add_argument("--model", default = "gpt2cgan")
    parser.add_argument("--buffer_size", default = 1000)
    parser.add_argument("--batch_size", default = 8)
    parser.add_argument("--epochs", default = 100)
    parser.add_argument("--noise_len", default = 2044)
    parser.add_argument("--noise_hidden_dim", default = 64)
    parser.add_argument("--example_to_generate", default = 16)
    parser.add_argument("--num_layer", default = 2)
    parser.add_argument("--num_head", default = 6)
    parser.add_argument("--num_last_dim", default = 500)
    parser.add_argument("--num_round", default = 3)
    parser.add_argument("--use_gpu", default = True)  
    parser.add_argument('--gpu_id', default = 0)
    parser.add_argument("--load_model", default = False)
    parser.add_argument("--model_path", default = None)              
    args = parser.parse_args()

    width = 20
    print("{0: <{width}}: {val}".format("mode", width = width, val = args.mode))
    print("{0: <{width}}: {val}".format("model", width = width, val = args.model))
    print("{0: <{width}}: {val}".format("buffer_size", width = width, val = args.buffer_size))
    print("{0: <{width}}: {val}".format("batch_size", width = width, val = args.batch_size))
    print("{0: <{width}}: {val}".format("epochs", width = width, val = args.epochs))
    print("{0: <{width}}: {val}".format("noise_len", width = width, val = args.noise_len))
    print("{0: <{width}}: {val}".format("noise_hidden_dim", width = width, val = args.noise_hidden_dim))
    print("{0: <{width}}: {val}".format("example_to_generate", width = width, val = args.example_to_generate))
    print("{0: <{width}}: {val}".format("num_layer", width = width, val = args.num_layer))
    print("{0: <{width}}: {val}".format("num_head", width = width, val = args.num_head))
    print("{0: <{width}}: {val}".format("num_last_dim", width = width, val = args.num_last_dim))
    print("{0: <{width}}: {val}".format("num_round", width = width, val = args.num_round))
    print("{0: <{width}}: {val}".format("use_gpu", width = width, val = args.use_gpu))
    print("{0: <{width}}: {val}".format("gpu_id", width = width, val = args.gpu_id))
    print("{0: <{width}}: {val}".format("load_model", width = width, val = args.load_model))
    print("{0: <{width}}: {val}".format("model_path", width = width, val = args.model_path))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


    # if use_gpu is TRUE, then get one and run the program on it
    if args.use_gpu:
        try:
            gpus = tf.config.list_physical_devices(device_type = 'GPU')
            print(gpus)
            
            if gpus:
                tf.config.set_visible_devices(devices = gpus[0], device_type = 'GPU')
        except:
            print("[No GPR] there is no availible gpu to use!!!")

    # get environment parameters
    now = datetime.now()
    time = now.strftime("%d_%m_%Y_%H_%M_%S")

    # get datsets
    # datasets, shape = initial_mnist_datset()
    # datasets = dataset_np(int(args.num_last_dim))
    datasets = load_dataset()

    if args.mode == "training":
        training(args = args, datasets = datasets, time = time)
    elif args.mode == "testing":
        pass
    elif args.mode == "find_vector":
        if args.model_path == None:
            error("Should provide model path to load")
        else:
            find_random_vector(model_path = args.model_path, model = str(args.model))
    else:
        pass

    