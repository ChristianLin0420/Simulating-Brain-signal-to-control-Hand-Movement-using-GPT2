
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from .gpt2cgan import gpt2cgan
from .classifier import get_pretrained_classfier, plot_spectrogram, stft_min_max
from utils.brain_activation import boolean_brain, transformation_matrix, restore_brain_activation_tf
from utils.model_monitor import record_model_weight

from sklearn.metrics import accuracy_score

class gpt2xcnn(tf.keras.Model):

    def __init__(self, config = None, generator = None, classifier = None, noise_len: int = 2089, noise_dim: int = 498, d_extra_steps: int = 5, last_dim: int = 3, **kwargs):

        super(gpt2xcnn, self).__init__()

        if generator is None:
            self.gptGenerator = gpt2cgan(config, noise_len = noise_len, noise_dim = noise_dim, d_extra_steps = d_extra_steps, last_dim = last_dim)
        else:
            self.gptGenerator = generator

        self.classifier = classifier

        (self.boolean_l, self.boolean_r) = boolean_brain()
        self.transformation_matrix = tf.constant(transformation_matrix())

        self.generated_count = 160

        # add one hot vector for every seed
        self.seed = tf.random.normal([self.generated_count, noise_len, noise_dim])

        tmp = [0] * int(self.generated_count / 2) + [1] * int(self.generated_count / 2)
        l = tf.constant(tmp)

        one_hot = tf.one_hot(l, depth = 2)
        one_hot = tf.expand_dims(one_hot, axis = 1)
        one_hot = tf.repeat(one_hot, repeats = noise_len, axis = 1)
        self.seed = tf.concat([self.seed, one_hot], axis = 2)

    def compile(self, optimizer, loss_fn):
        super(gpt2xcnn, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):

        seeds, labels = data

        signals = tf.constant([])
        generate_count = 32
        generate_round = int(seeds.shape[0] / generate_count)

        # rgb_weights = tf.constant([0.2989, 0.5870, 0.1140], shape=[3, 1])

        loss = tf.constant([])
        Y_pred = None

        for idx in range(generate_round):
            
            print("idx: {}, shape of seeds: {}".format(idx, seeds[idx * generate_count:(idx + 1) * generate_count].shape))

            with tf.GradientTape() as tape:
                sigs = self.gptGenerator.generator(seeds[idx * generate_count:(idx + 1) * generate_count])

                brain = None
                signals = None

                for i in range(sigs.shape[0]):
                    (l_tmp, r_tmp) = restore_brain_activation_tf(sigs[i], self.boolean_l, self.boolean_r)
                    brain_activation = tf.concat([l_tmp, r_tmp], axis = 0)
                    brain_activation = tf.expand_dims(brain_activation, axis = 0)

                    if brain == None:
                        brain = brain_activation
                    else:
                        brain = tf.concat([brain, brain_activation], axis = 0)

                brain = tf.reshape(brain, shape = [brain.shape[0], brain.shape[1], brain.shape[2]])

                for i in range(brain.shape[0]):
                    signal = tf.matmul(self.transformation_matrix, brain[i])
                    signal = tf.expand_dims(signal, axis = 0)

                    if signals == None:
                        signals = signal
                    else:
                        signals = tf.concat([signals, signal], axis = 0)

                signals = signals[:, 11:14, :]

                X = stft_min_max(signals)

                y_pred = self.classifier(X)
                y_true = labels[idx * generate_count: (idx + 1) * generate_count]

                loss = self.loss_fn(y_true, y_pred)

                if Y_pred == None:
                    Y_pred = y_pred
                else:
                    Y_pred = tf.concat([Y_pred, y_pred], axis = 0)
            
            grads = tape.gradient(loss, self.gptGenerator.generator.trainable_weights)
            #print("grads: {}".format(grads))
            tf.print(grads)
            self.optimizer.apply_gradients(zip(grads, self.gptGenerator.generator.trainable_weights))

        Y_pred = tf.reshape(Y_pred, [Y_pred.shape[0]])

        labels = tf.cast(labels, tf.int64)
        Y_pred = tf.cast(Y_pred, tf.int64)

        accuracy = tf.math.equal(labels, Y_pred)
        accuracy = tf.math.reduce_mean(tf.cast(accuracy, tf.float32))

        print("start generating new signals")

        # generate image from given seed
        predictions = None

        for i in range(10):
            prediction = self.gptGenerator.generator(self.seed[i * self.generate_count : (i + 1) * self.generate_count], training = False)
            
            if predictions == None:
                predictions = prediction
            else:
                predictions = tf.concat([predictions, prediction], axis = 0)


        return {"loss": loss, "accuracy": accuracy, "generated": predictions}

        # return {"loss": loss, "accuracy": acc}