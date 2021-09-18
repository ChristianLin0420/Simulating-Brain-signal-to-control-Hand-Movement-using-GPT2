

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
from utils.datasetGenerator import DatasetGenerator, get_training_filenames_and_labels, get_training_raw_signals, generate_random_vaectors
from utils.callback import EarlyStoppingAtMinLoss, RecordGeneratedImages
from utils.model_monitor import save_loss_range_record, save_loss_record, save_random_vector, save_result_as_gif, show_generated_image

from model.gpt2gan import gpt2gan
from model.gpt2wgan import gpt2wgan
from model.gpt2cgan import gpt2cgan
from model.gpt2xCNN import gpt2xcnn
from model.model_utils import load_model
from config.config_gpt2 import GPT2Config, save_model_config

LEARNING_RATE = 0.0003


def training(args, datasets, time, num_classes: int = 2):
    
    # initial model
    config = GPT2Config(n_layer = int(args.num_layer), 
                        n_head = int(args.num_head), 
                        n_embd = int(args.noise_hidden_dim), 
                        n_positions = int(args.noise_len))

    g_loss_collection = []
    d_loss_collection = []

    loss_collection = []
    acc_collection = []

    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    round_num = int(args.num_round)
    last_dim = int(args.num_last_dim)
    subject_count = int(args.subject_count)

    current_round = 1
    add_class_dim = False

    while round_num >= current_round:

        print("\n--------------------------------------------------------------")
        print("-------------------------- Round {} --------------------------".format(current_round))
        print("--------------------------------------------------------------\n")

        d_optimizer = keras.optimizers.Adam(learning_rate = LEARNING_RATE)
        g_optimizer = keras.optimizers.Adam(learning_rate = LEARNING_RATE)
        optimizer = keras.optimizers.Adam(learning_rate = LEARNING_RATE)
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
            
            ROOT_DIR = '/home/jupyter-ivanljh123/rsc/Source Estimate'
            _, dirs, _ = os.walk(ROOT_DIR).__next__()

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

            filenames, labels = get_training_filenames_and_labels(batch_size = batch_size, subject_count = subject_count)
            raw_filenames, raw_labels = get_training_raw_signals(subject_count = subject_count)
            dataGenerator = DatasetGenerator(filenames = filenames, raw_filenames = raw_filenames, labels = labels, raw_labels = raw_labels, batch_size = batch_size, subject_count = subject_count)

            train_x = np.asarray([])
            train_y = np.asarray([])

            eye_close_data = np.asarray([])
            eye_open_data = np.asarray([])

            (eye_close_data, eye_open_data) = dataGenerator.get_event()
            (raw_x, raw_y) = dataGenerator.get_raw()

            print("raw_x shape: {}".format(raw_x.shape))
            print("raw_y shape: {}".format(raw_y.shape))

            for idx in range(subject_count):

                print("\n================================================= Subject {} =================================================\n".format(idx))
                
                get_data = False
                (train_x, train_y, get_data) = dataGenerator.getItem()

                while not get_data:
                    (train_x, train_y, valid) = dataGenerator.getItem()

                    if valid:
                        get_data = True
                        break

                print("train_x shape: {}".format(train_x.shape))
                print("train_y shape: {}".format(train_y.shape))

                history = model.fit(
                            x = train_x,
                            y = train_y,
                            batch_size = batch_size,
                            epochs = epochs, 
                            verbose = 1, 
                            callbacks = [RecordGeneratedImages(time, current_round, args.model, eye_close_data, eye_open_data, raw_x, raw_y)]#, tensorboard_callback]
                        )

                g_loss = history.history['g_loss']
                d_loss = history.history['d_loss']

                g_loss_collection.append(g_loss)
                d_loss_collection.append(d_loss)

                # save training loss figure
                save_loss_record(np.arange(1, len(g_loss) + 1), g_loss, d_loss, time, str(args.model), current_round)
                save_loss_range_record(np.arange(len(g_loss_collection[0])), g_loss_collection, time, args.model, "g_loss")
                save_loss_range_record(np.arange(len(d_loss_collection[0])), d_loss_collection, time, args.model, "d_loss")

                g_loss = None
                d_loss = None
                history = None

                del g_loss
                del d_loss
                del history

            # save model
            save_model_config(config, str(args.model), time, current_round)
            model.save_weights("./trained_model/" + str(args.model) + "/" + str(time) + "/model_" + str(current_round), save_format = 'tf')

        elif args.model == "gpt2xcnn":
            ROOT_DIR = '/home/jupyter-ivanljh123/rsc/Source Estimate'
            _, dirs, _ = os.walk(ROOT_DIR).__next__()

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

            filenames, labels = get_training_filenames_and_labels(batch_size = batch_size, subject_count = subject_count)
            raw_filenames, raw_labels = get_training_raw_signals(subject_count = subject_count)
            dataGenerator = DatasetGenerator(filenames = filenames, raw_filenames = raw_filenames, labels = labels, raw_labels = raw_labels, batch_size = batch_size, subject_count = subject_count)

            train_x = np.asarray([])
            train_y = np.asarray([])

            eye_close_data = np.asarray([])
            eye_open_data = np.asarray([])

            (eye_close_data, eye_open_data) = dataGenerator.get_event()
            (raw_x, raw_y) = dataGenerator.get_raw()

            print("raw_x shape: {}".format(raw_x.shape))
            print("raw_y shape: {}".format(raw_y.shape))

            for idx in range(subject_count):

                print("\n================================================= Subject {} =================================================\n".format(idx))
                
                get_data = False
                (train_x, train_y, get_data) = dataGenerator.getItem()

                while not get_data:
                    (train_x, train_y, valid) = dataGenerator.getItem()

                    if valid:
                        get_data = True
                        break

                print("train_x shape: {}".format(train_x.shape))
                print("train_y shape: {}".format(train_y.shape))

                history = model.fit(
                            x = train_x,
                            y = train_y,
                            batch_size = batch_size,
                            epochs = epochs, 
                            verbose = 1, 
                            callbacks = [RecordGeneratedImages(time, current_round, args.model, eye_close_data, eye_open_data, raw_x, raw_y)]#, tensorboard_callback]
                        )

                g_loss = history.history['g_loss']
                d_loss = history.history['d_loss']

                g_loss_collection.append(g_loss)
                d_loss_collection.append(d_loss)

                # save training loss figure
                save_loss_record(np.arange(1, len(g_loss) + 1), g_loss, d_loss, time, str(args.model), current_round)
                save_loss_range_record(np.arange(len(g_loss_collection[0])), g_loss_collection, time, args.model, "g_loss")
                save_loss_range_record(np.arange(len(d_loss_collection[0])), d_loss_collection, time, args.model, "d_loss")

                g_loss = None
                d_loss = None
                history = None

                del g_loss
                del d_loss
                del history

            # save model
            save_model_config(config, str(args.model), time, current_round)
            model.save_weights("./trained_model/" + str(args.model) + "/" + str(time) + "/model_" + str(current_round), save_format = 'tf')


            # using trained model with classifier

            random_vectors, labels = generate_random_vaectors()

            new_model = gpt2xcnn(generator = model)

            new_model.compile(
                optimizer = optimizer,
                loss_fn = loss_fn
            )

            new_history = new_model.fit(
                            x = random_vectors, 
                            y = labels, 
                            batch_size = 64, 
                            epochs = epochs, 
                            verbose = 1 )
            

            loss = new_history.history['loss']
            acc  = new_history.history['accuracy']

            loss_collection.append(loss)
            acc_collection.append(acc)

            # save model
            new_model.save_weights("./trained_model/" + str(args.model) + "/" + str(time) + "/model_" + str(current_round) + "_with_classifier", save_format = 'tf')


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
    elif args.model == "gpt2xcnn":
        save_loss_range_record(np.arange(len(loss_collection)), loss_collection, time, args.model, "loss")
        save_loss_range_record(np.arange(len(loss_collection)), acc_collection, time, args.model, "accuracy")
        

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
    parser.add_argument("--epochs", default = 50)
    parser.add_argument("--noise_len", default = 2089) 
    parser.add_argument("--noise_hidden_dim", default = 498)
    parser.add_argument("--example_to_generate", default = 16)
    parser.add_argument("--num_layer", default = 2)
    parser.add_argument("--num_head", default = 5)
    parser.add_argument("--num_last_dim", default = 1)
    parser.add_argument("--num_round", default = 3)
    parser.add_argument("--subject_count", default = 1)
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
    print("{0: <{width}}: {val}".format("subject_count", width = width, val = args.subject_count))
    print("{0: <{width}}: {val}".format("use_gpu", width = width, val = args.use_gpu))
    print("{0: <{width}}: {val}".format("gpu_id", width = width, val = args.gpu_id))
    print("{0: <{width}}: {val}".format("load_model", width = width, val = args.load_model))
    print("{0: <{width}}: {val}".format("model_path", width = width, val = args.model_path))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    np.seterr(all="ignore")

    # if use_gpu is TRUE, then get one and run the program on it
    if args.use_gpu:
        try:
            gpus = tf.config.list_physical_devices(device_type = 'GPU')
            
            if gpus:
                tf.config.set_visible_devices(devices = gpus[0], device_type = 'GPU')
                tf.config.experimental.set_memory_growth(gpus[0], True)
                # tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=23000)])
        except:
            print("[No GPR] there is no availible gpu to use!!!")

    # get environment parameters
    now = datetime.now()
    time = now.strftime("%d_%m_%Y_%H_%M_%S")

    if args.mode == "training":
        training(args = args, datasets = None, time = time)
    elif args.mode == "find_vector":
        if args.model_path == None:
            error("Should provide model path to load")
        else:
            find_random_vector(model_path = args.model_path, model = str(args.model))
    else:
        pass

    