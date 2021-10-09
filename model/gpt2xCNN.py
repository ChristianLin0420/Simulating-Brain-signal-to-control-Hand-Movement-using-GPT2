
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from .gpt2cgan import gpt2cgan
from .classifier import get_pretrained_classfier, plot_spectrogram
from utils.brain_activation import boolean_brain, transformation_matrix, restore_brain_activation_tf
from utils.model_monitor import record_model_weight

from sklearn.metrics import accuracy_score

class gpt2xcnn(tf.keras.Model):

    def __init__(self, config = None, generator = None, classifier = None, noise_len: int = 784, noise_dim: int = 32, d_extra_steps: int = 5, last_dim: int = 3, **kwargs):

        super(gpt2xcnn, self).__init__()

        if generator is None:
            self.gptGenerator = gpt2cgan(config, noise_len = noise_len, noise_dim = noise_dim, d_extra_steps = d_extra_steps, last_dim = last_dim)
        else:
            self.gptGenerator = generator

        self.classifier = classifier

        (self.boolean_l, self.boolean_r) = boolean_brain()
        self.transformation_matrix = tf.constant(transformation_matrix())

    def compile(self, optimizer, loss_fn):
        super(gpt2xcnn, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):

        seeds, labels = data

        signals = tf.constant([])
        generate_count = 4
        generate_round = int(seeds.shape[0] / generate_count)

        # print("generate_count: {}".format(generate_count))
        # print("generate_round: {}".format(generate_round))

        # left_brain_activation = tf.constant([])
        # right_brain_activation = tf.constant([])

        rgb_weights = tf.constant([0.2989, 0.5870, 0.1140], shape=[3, 1])

        # images = tf.constant([])

        loss = tf.constant([])
        Y_pred = None

        for idx in range(generate_round):
            
            # print("idx: {}, shape of seeds: {}".format(idx, seeds[idx * generate_count:(idx + 1) * generate_count].shape))

            with tf.GradientTape() as tape:
                sigs = self.gptGenerator.generator(seeds[idx * generate_count:(idx + 1) * generate_count])

                # print("idx: {}, shape of sigs: {}".format(idx, sigs.shape))

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

                # brain_activation = tf.concat([l_tmp, r_tmp], axis = 0)
                # print("brain shape: {}".format(brain.shape))
                brain = tf.reshape(brain, shape = [brain.shape[0], brain.shape[1], brain.shape[2]])
                # print("brain_activation shape: {}".format(brain.shape))
                # print("transformation_matrix shape: {}".format(self.transformation_matrix.shape))

                for i in range(brain.shape[0]):
                    signal = tf.matmul(self.transformation_matrix, brain[i])
                    signal = tf.expand_dims(signal, axis = 0)

                    if signals == None:
                        signals = signal
                    else:
                        signals = tf.concat([signals, signal], axis = 0)

                signals = signals[:, 11:14, :]
                # print("signal shape: {}".format(signals.shape))

                Zxx = None

                for i in range(signals.shape[0]):
                    zxx = tf.signal.stft(signals[i], frame_length = 256, frame_step = 16)
                    zxx = tf.abs(zxx)

                    # print("images shape: {}".format(zxx.shape))
                    # print("images type: {}".format(type(zxx)))
                    zxx = zxx[:, :, :40]
                    zxx = tf.expand_dims(zxx, axis = 0)

                    if Zxx == None:
                        Zxx = zxx
                    else:
                        Zxx = tf.concat([Zxx, zxx], axis = 0)
                
                # print("Zxx shape: {}".format(Zxx.shape))

                images = None

                for k in range(Zxx.shape[0]):
                    current_image = None

                    for i in range(Zxx[k].shape[0]):
                        if current_image == None:
                            current_image = Zxx[k][i]
                        else:
                            current_image = tf.concat([current_image, Zxx[k][i]], axis = 0)

                    current_image = tf.expand_dims(current_image, axis = 0)

                    if images == None:
                        images = current_image
                    else:
                        images = tf.concat([images, current_image], axis = 0)

                # print("images shape: {}".format(images.shape))

                X = images
                # Expand last dimension to gray scale
                X = tf.expand_dims(X, axis = 3)

                # print("Input X shape: {}".format(X.shape))

                y_pred = self.classifier(X)
                y_true = labels[idx * generate_count: (idx + 1) * generate_count]
                # print("y_pred shape: {}".format(y_pred.shape))
                # print("y_true shape: {}".format(y_true.shape))
                loss = self.loss_fn(y_true, y_pred)

                # print("y_pred: {}".format(y_pred))
                # print("y_pred type: {}".format(type(y_pred)))
                # print("loss: {}".format(loss))

                if Y_pred == None:
                    Y_pred = y_pred
                else:
                    Y_pred = tf.concat([Y_pred, y_pred], axis = 0)
            
            grads = tape.gradient(loss, self.gptGenerator.generator.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.gptGenerator.generator.trainable_weights))

            # record the weight to observe the change of the weight
            record_model_weight(self.gptGenerator.generator.trainable_weights)

        Y_pred = tf.reshape(Y_pred, [Y_pred.shape[0]])

        labels = tf.cast(labels, tf.int64)
        Y_pred = tf.cast(Y_pred, tf.int64)

        # print("labels : {}".format(labels))
        # print("Y_pred : {}".format(Y_pred))

        # print("labels shape: {}".format(labels.shape))
        # print("Y_pred shape: {}".format(Y_pred.shape))

        # print("labels type: {}".format(type(labels)))
        # print("Y_pred type: {}".format(type(Y_pred)))

        accuracy = tf.math.equal(labels, Y_pred)
        accuracy = tf.math.reduce_mean(tf.cast(accuracy, tf.float32))

        # print("loss: {}".format(loss))
        # print("accuracy: {}".format(accuracy))

        return {"loss": loss, "accuracy": accuracy}

        # return {"loss": loss, "accuracy": acc}