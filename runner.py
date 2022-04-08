
import io
import json
import numpy as np

from tensorflow import keras
from tensorflow.keras.optimizers import Adam

from logging import error

from modules.gpt2cgan import gpt2cgan
from modules.gpt2xcnn import gpt2xcnn
from modules.gpt2sgan import gpt2sgan

from modules.classifier import (
    get_pretrained_classfier_from_path, 
)

from utils.callback import (
    # EarlyStoppingAtMinLoss, 
    # RecordGeneratedImages, 
    # RecordWeight, 
    # RecordReconstructedGeneratedImages, 
    Accuracy, 
    Loss, 
    GANLoss,
    STFTgenerator
)

from utils.datasetGenerator import (
    DatasetGenerator,
    generate_random_vectors, 
    get_training_raw_signals, 
    get_training_reconstruct_signals
)


class Runner():

    def __init__(self, config, time):
        self.config = config
        self.time = time

        ## pre-settings
        self.get_training_dataset()

        ## pretrained classifier setting(for fine-tuning)
        if config.fine_tune:
            self.classifier = get_pretrained_classfier_from_path(config.pretrained_classifier_path)
            self.c_optimizer = Adam(learning_rate = config.learning_rate)
            self.classifier.compile(optimizer = self.c_optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])

        ## generator and discriminator parameters setting
        self.d_optimizer = keras.optimizers.Adam(learning_rate = config.learning_rate)
        self.g_optimizer = keras.optimizers.Adam(learning_rate = config.learning_rate)
        self.optimizer = keras.optimizers.Adam(learning_rate = config.learning_rate)
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits = False)
        self.loss_kl = keras.losses.KLDivergence()
    
    def get_training_dataset(self):
        ### retrieve training data
        filenames, labels = get_training_reconstruct_signals()
        raw_filenames, raw_labels = get_training_raw_signals(subject_count = self.config.subject_count)
        dataGenerator = DatasetGenerator(filenames = filenames, raw_filenames = raw_filenames, labels = labels, raw_labels = raw_labels, config = self.config)
        
        (self.train_x, self.train_y, self.real_average_data) = dataGenerator.get_reconstructed_items(filenames, labels)

        self.left_hand_data = np.expand_dims(self.real_average_data[0], axis = 0)
        self.right_hand_data = np.expand_dims(self.real_average_data[1], axis = 0)

    def get_model(self):
        ## model initialization
        if self.config.model_name == "gpt2cgan":
            model = gpt2cgan(
                config = self.config, 
                data_avg = self.real_average_data
            )
            
            ## compile the model
            model.compile(
                d_optimizer = self.d_optimizer,
                g_optimizer = self.g_optimizer,
                loss_fn = self.loss_fn,
                loss_kl = self.loss_kl
            )

            return model
        elif (self.config.model_name == "gpt2xcnn") or (self.config.model_name == "gpt2sgan"):
            pretrained_model = gpt2cgan(
                data_avg = self.real_average_data, 
                config = self.config
            )

            ## fine-tune (load model)
            if self.config.fine_tune:
                pretrained_model.load_weights(self.config.pretrained_finetune_path)
                pretrained_model.trainable = True

            pretrained_model.compile(
                d_optimizer = self.d_optimizer,
                g_optimizer = self.g_optimizer,
                loss_fn = self.loss_fn,
                loss_kl = self.loss_kl
            )

            ## fine-tune (build new model)
            if self.config.model_name == "gpt2xcnn":
                model = gpt2xcnn(
                    data_avg = self.real_average_data, 
                    config = self.config, 
                    generator = pretrained_model, 
                    classifier = self.classifier
                )
            else:
                model = gpt2sgan(
                    data_avg = self.real_average_data, 
                    config = self.config, 
                    generator = pretrained_model, 
                    classifier = self.classifier
                )
        
            ## compile the model
            model.compile(
                optimizer = self.optimizer,
                loss_fn = self.loss_fn,
                loss_kl = self.loss_kl
            )
            return model
        else:
            error("[Runner] invalid model name was given!!!")
            return None

    def set_callbacks(self, _round):
        
        callbacks = list()

        if self.config.Accuracy:
            self.callbacks.append(Accuracy(self.config, self.time, _round))

        if self.config.Loss:
            self.callbacks.append(Loss(self.config, self.time, _round))

        if self.config.STFTgenerator:
            self.callbacks.append(STFTgenerator(self.config, self.time, _round))

        return callbacks

    def store_history(self, history, _round):
        for key in list(history.history.keys()):
            if key not in ["generated", "accuracy", "loss"]:
                value = np.asarray(history.history[str(key)]).tolist()

                data = { str(key) : value }
                with io.open("result/{}/{}/history/{}/{}.json".format(self.config.model_name, self.time, _round, key), 'w', encoding = 'utf8') as outfile:
                    s = json.dumps(data, indent = 4, sort_keys = True, ensure_ascii = False)
                    outfile.write(s)

    def run(self):

        ## initialize random vectors
        random_vectors, random_vectors_labels, _ = generate_random_vectors(     num = self.config.random_vector_num,
                                                                                length = self.config.n_positions, 
                                                                                emb = self.config.n_embd, 
                                                                                class_rate_random_vector = self.config.class_rate_random_vector, 
                                                                                class_count = self.config.class_count,
                                                                                variance = self.config.noise_variance   )
        
        ## start training
        for idx in range(self.config.rounds):

            ## set required callbacks
            model = self.get_model()
            callbacks = self.set_callbacks(idx)

            ## start training and record training histroy
            if self.config.model_name == "gpt2cgan":
                history = model.fit(x = self.train_x,
                                    y = self.train_y,
                                    batch_size = self.config.batch_size,
                                    epochs = self.config.epochs, 
                                    verbose = 1
                                )
            elif self.config.model_name == "gpt2xcnn":
                history = model.fit(x = random_vectors, 
                                    y = random_vectors_labels, 
                                    batch_size = self.config.batch_size, 
                                    epochs = self.config.epochs, 
                                    verbose = 1,
                                    callbacks = callbacks
                                )
            elif self.config.model_name == "gpt2sgan":
                history = model.fit(x = random_vectors, 
                                    y = random_vectors_labels, 
                                    batch_size = self.config.batch_size, 
                                    epochs = self.config.epochs, 
                                    verbose = 1,
                                    callbacks = callbacks
                                )
            else:
                error("[Runner] invalid training model!!!")
                return
            
            ## store training history
            self.store_history(history, idx)

            ## save model
            model.save_weights("result/{}/{}/models/{}/model".format(self.config.model_name, self.time, idx), save_format = 'tf')