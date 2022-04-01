
import io
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from .gpt2cgan import gpt2cgan
from .classifier import get_pretrained_classfier, plot_spectrogram, stft_min_max
from utils.brain_activation import boolean_brain, restore_brain_activation, transformation_matrix, restore_brain_activation_tf
from utils.model_monitor import record_model_weight

from sklearn.metrics import accuracy_score

class gpt2xcnn(tf.keras.Model):

    def __init__(self, data_avg = None, config = None, generator = None, classifier = None, d_extra_steps: int = 5, last_dim: int = 3, **kwargs):

        super(gpt2xcnn, self).__init__()

        if generator is None:
            self.gptGenerator = gpt2cgan(config, data_avg, noise_len = config.n_positions, noise_dim = config.n_embd, d_extra_steps = d_extra_steps, last_dim = last_dim)
        else:
            self.gptGenerator = generator

        self.classifier = classifier

        (self.boolean_l, self.boolean_r) = boolean_brain()
        self.transformation_matrix = tf.constant(transformation_matrix())

        self.generate_count = 32

        # add one hot vector for every seed
        self.seed = tf.random.normal([self.generate_count, config.n_positions, config.n_embd])

        tmp = [0] * int(self.generate_count / 2) + [1] * int(self.generate_count / 2)
        l = tf.constant(tmp)

        one_hot = tf.one_hot(l, depth = 2)
        one_hot = tf.expand_dims(one_hot, axis = 1)
        one_hot = tf.repeat(one_hot, repeats = config.n_positions, axis = 1)
        self.seed = tf.concat([self.seed, one_hot], axis = 2)

        self.epoch_count = 0
        self.epoch_acc_average = None

        self.t_matrix = tf.constant(transformation_matrix(), dtype = tf.float32)
        (self.boolean_l, self.boolean_r) = boolean_brain()

        self.real_data = tf.constant(data_avg, dtype = tf.float32)
        self.real_data = tf.reshape(self.real_data, shape = [self.real_data.shape[0], self.real_data.shape[1], self.real_data.shape[2], 1])

        self.delta = 0.0001
        self.count = 0

    def compile(self, optimizer, loss_fn, loss_kl):
        super(gpt2xcnn, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_kl = loss_kl

    def set_weights(self, weights):
        tf.print("weight: {}".format(weights))
        self.count = weights

    def train_step(self, data):

        seeds, labels = data

        signals_stft = tf.constant([])
        generate_count = 4
        generate_round = int(seeds.shape[0] / generate_count)

        tt = self.optimizer.weights

        loss = tf.constant([])
        Y_pred = None
        XX = None

        sigs_spectrum = tf.constant([])
        sigs_record = tf.constant([])

        idx = 0
        tmp_delta = 0.0
            
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
            signals_stft = signals

            X = stft_min_max(signals)

            if XX is None:
                XX = X
            else:
                XX = tf.concat([XX, X], axis = 0)


            y_pred = self.classifier(X)
            print(type(y_pred))
            tmp = []
            for pred in y_pred:
                if pred[0] > pred[1]:
                    tmp.append(0)
                else:
                    tmp.append(1)
            y_pred = tmp
            y_true = labels[idx * generate_count: (idx + 1) * generate_count]

            loss = self.loss_fn(y_true, y_pred)

            if Y_pred == None:
                Y_pred = y_pred
            else:
                Y_pred = tf.concat([Y_pred, y_pred], axis = 0)
        
            # grads = tape.gradient(loss, self.gptGenerator.generator.trainable_weights)
            # self.optimizer.apply_gradients(zip(grads, self.gptGenerator.generator.trainable_weights))

            Y_pred = tf.reshape(Y_pred, [Y_pred.shape[0]])

            labels = tf.cast(labels, tf.int64)
            Y_pred = tf.cast(Y_pred, tf.int64)

            accuracy = tf.math.equal(labels, Y_pred)
            accuracy = tf.math.reduce_mean(tf.cast(accuracy, tf.float32))
            accuracy = tf.expand_dims(accuracy, axis = 0)

            # generate image from given seed
            predictions_left = None
            predictions_right = None
            generate_count = 8
            generate_round = int(self.generate_count / (generate_count * 2))

            for i in range(generate_round):
                predictions_l = self.gptGenerator.generator(self.seed[i * generate_count : (i + 1) * generate_count], training = False)
                predictions_r = self.gptGenerator.generator(self.seed[(i + generate_round) * generate_count : (i + generate_round + 1) * generate_count], training = False)

                if predictions_left == None:
                    predictions_left = predictions_l
                else:
                    predictions_left = tf.concat([predictions_left, predictions_l], axis = 0)

                if predictions_right == None:
                    predictions_right = predictions_r
                else:
                    predictions_right = tf.concat([predictions_right, predictions_r], axis = 0)

            print("predictions_left shape: {}".format(predictions_left.shape))
            print("predictions_right shape: {}".format(predictions_right.shape))

            predictions_left = tf.reduce_mean(predictions_left, axis = 0)
            predictions_right = tf.reduce_mean(predictions_right, axis = 0)

            predictions_left = tf.reshape(predictions_left, shape = [1, predictions_left.shape[0], predictions_left.shape[1], predictions_left.shape[2]])
            predictions_right = tf.reshape(predictions_right, shape = [1, predictions_right.shape[0], predictions_right.shape[1], predictions_right.shape[2]])

            predictions = tf.concat([predictions_left, predictions_right], axis = 0)
            predictions = tf.concat([predictions, self.real_data], axis = 0)

            print("predictions shape: {}".format(predictions.shape))

            ## raw signal kl
            data1 = tf.reshape(self.real_data[0], shape = [1, 2089, 500])
            data2 = tf.reshape(self.real_data[1], shape = [1, 2089, 500])
            raw_signal_feet_kl = self.loss_kl(data1, predictions_left)
            raw_signal_tongue_kl = self.loss_kl(data2, predictions_right)

            ## spectrum kl
            sigs = predictions
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
                signal = tf.matmul(self.t_matrix, brain[i])
                signal = tf.expand_dims(signal, axis = 0)

                if signals == None:
                    signals = signal
                else:
                    signals = tf.concat([signals, signal], axis = 0)

            signals = signals[:, 11:14, :]

            Zxx = tf.signal.stft(signals, frame_length=256, frame_step=16)
            Zxx = tf.abs(Zxx)

            print("Zxx shape: {}".format(Zxx.shape))

            raw_feet_spectrum_c3_signal_kl = self.loss_kl(signals[2, 0, :], signals[0, 0, :])
            raw_tongue_spectrum_c3_signal_kl = self.loss_kl(signals[3, 0, :], signals[1, 0, :])
            raw_feet_spectrum_c4_signal_kl = self.loss_kl(signals[2, 1, :], signals[0, 1, :])
            raw_tongue_spectrum_c4_signal_kl = self.loss_kl(signals[3, 1, :], signals[1, 1, :])
            raw_feet_spectrum_cz_signal_kl = self.loss_kl(signals[2, 2, :], signals[0, 2, :])
            raw_tongue_spectrum_cz_signal_kl = self.loss_kl(signals[3, 2, :], signals[1, 2, :])

            orignal_loss = loss
            sum_kl = tf.math.add(raw_signal_feet_kl, raw_signal_tongue_kl)
            delta_kl_loss = tf.math.multiply(tf.constant(-self.delta), sum_kl)
            loss = loss + delta_kl_loss
            
            # tf.print("=" * 30)
            # tf.print("delta: {}".format(tf.constant(-self.delta)))
            # tf.print("delta_kl_loss prev: {}, later: {}".format(delta_kl_loss, sum_kl))
            # tf.print("=" * 30)

        Zxx = None
        X = None
        sigs = None
        brain = None
        signals = None
        data1 = None
        data2 = None


        del Zxx
        del X
        del sigs
        del brain
        del signals
        del data1
        del data2

        # update gradient
        grads = tape.gradient(orignal_loss, self.gptGenerator.generator.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.gptGenerator.generator.trainable_weights))


        return {"loss": loss, 
                "accuracy": accuracy, 
                "raw_feet_signal_kl": raw_signal_feet_kl, 
                "raw_tongue_signal_kl": raw_signal_tongue_kl, 
                "raw_feet_spectrum_c3_signal_kl": raw_feet_spectrum_c3_signal_kl, 
                "raw_tongue_spectrum_c3_signal_kl": raw_tongue_spectrum_c3_signal_kl, 
                "raw_feet_spectrum_c4_signal_kl": raw_feet_spectrum_c4_signal_kl, 
                "raw_tongue_spectrum_c4_signal_kl": raw_tongue_spectrum_c4_signal_kl, 
                "raw_feet_spectrum_cz_signal_kl": raw_feet_spectrum_cz_signal_kl, 
                "raw_tongue_spectrum_cz_signal_kl": raw_tongue_spectrum_cz_signal_kl, 
                "generated": predictions, 
                "original_loss": orignal_loss, 
                "delta_kl_loss": delta_kl_loss, 
                "sum_kl": sum_kl, 
                "delta": self.delta}

    def call(self, inputs):
        
        predictions = np.asarray(self.generator(inputs, training = False))

        batch = predictions.shape[0]

        filtered_activations = np.asarray([])

        for idx in range(batch):
            (l_act, r_act) = restore_brain_activation(predictions[idx], self.boolean_l, self.boolean_r)
            act = np.concatenate([l_act, r_act], axis = 0)
            eeg_signal = np.dot(self.t_matrix, act)
            eeg_signal = np.expand_dims(eeg_signal, axis = 0)

            motor_signal = np.asarray([])

            for name, i in enumerate(eeg_signal):
                if name in CHANNEL_NAME:
                    sig = np.expand_dims(eeg_signal[i], axis = 0)

                    if len(motor_signal) == 0:
                        motor_signal = sig
                    else:
                        motor_signal = np.concatenate([motor_signal, sig], axis = 0)

            if len(filtered_activations) == 0:
                filtered_activations = motor_signal
            else:
                filtered_activations = np.concatenate([filtered_activations, motor_signal], axis = 0)

        print("filtered activation shape: {}".format(filtered_activations.shape))
            
        return filtered_activations