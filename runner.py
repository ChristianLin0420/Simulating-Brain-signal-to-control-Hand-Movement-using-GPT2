
from logging import error
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.optimizers import Adam

from modules.gpt2gan import gpt2gan
from modules.gpt2wgan import gpt2wgan
from modules.gpt2cgan import gpt2cgan
from modules.gpt2xCNN import gpt2xcnn

from modules.classifier import (
    get_pretrained_classfier, 
    get_pretrained_classfier_from_path, 
    stft_min_max
)

from utils.callback import (
    EarlyStoppingAtMinLoss, 
    RecordGeneratedImages, 
    RecordWeight, 
    RecordReconstructedGeneratedImages
)


class Runner():

    def __init__(self, config):
        self.config = config

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

        ## generator initialization
        if config.model_name == "gpt2cgan":
            self.model = gpt2cgan(config = config)
        elif config.model_name == "gpt2xcnn":
            self.model = gpt2xcnn(config = config)
        else:
            error("[Runner] invalid model name was given!!!")


        ## discriminator initialization

    def run(self):
        pass

    # def 