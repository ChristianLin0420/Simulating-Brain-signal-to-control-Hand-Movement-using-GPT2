
import numpy as np
import tensorflow as tf

from res.brain import Brain
from .gpt2cgan import gpt2cgan
from .classifier import stft_min_max
from utils.brain_activation import boolean_brain, restore_brain_activation, transformation_matrix, restore_brain_activation_tf
from utils.datasetGenerator import generate_random_vectors

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

        self.generate_count = config.example_to_generate
        assert self.generate_count % config.class_count == 0

        # add one hot vector for every seed
        self.seed, _, self.sub_vector_size = generate_random_vectors(self.generate_count, config.n_positions, config.n_embd, config.class_rate_random_vector, config.class_count, config.noise_variance, False)

        self.t_matrix = tf.constant(transformation_matrix(), dtype = tf.float32)
        (self.boolean_l, self.boolean_r) = boolean_brain()

        self.real_data = tf.constant(data_avg, dtype = tf.float32)
        self.real_data = tf.reshape(self.real_data, shape = [self.real_data.shape[0], self.real_data.shape[1], self.real_data.shape[2], 1])

        self.delta = 0.0001
        self.config = config

    def compile(self, optimizer, loss_fn, loss_kl):
        super(gpt2xcnn, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_kl = loss_kl

    def set_weights(self, weights):
        tf.print("weight: {}".format(weights))

    def train_step(self, data):
        
        seeds, labels = data
        labels = tf.cast(labels, tf.float32)
        labels = tf.reshape(labels, [self.config.batch_size, 1])

        loss = tf.constant([])
        Y_pred = None

        with tf.GradientTape() as tape:
            sigs = self.gptGenerator.generator(seeds)

            brain = None
            signals = None

            # restore the generated signals back to source activity
            for i in range(sigs.shape[0]):
                (l_tmp, r_tmp) = restore_brain_activation_tf(sigs[i], self.boolean_l, self.boolean_r)
                brain_activation = tf.concat([l_tmp, r_tmp], axis = 0)
                brain_activation = tf.expand_dims(brain_activation, axis = 0)

                if brain == None:
                    brain = brain_activation
                else:
                    brain = tf.concat([brain, brain_activation], axis = 0)

            brain = tf.reshape(brain, shape = [brain.shape[0], brain.shape[1], brain.shape[2]])

            # inverse procedure
            for i in range(brain.shape[0]):
                signal = tf.matmul(self.transformation_matrix, brain[i])
                signal = tf.expand_dims(signal, axis = 0)

                if signals == None:
                    signals = signal
                else:
                    signals = tf.concat([signals, signal], axis = 0)

            signals = signals[:, 11:14, :]
            X = stft_min_max(signals)

            # training records (loss, accuracy)
            y_pred = self.classifier(X)
            loss = self.loss_fn(labels, y_pred)

            if Y_pred == None:
                Y_pred = y_pred
            else:
                Y_pred = tf.concat([Y_pred, y_pred], axis = 0)

            Y_pred = tf.reshape(Y_pred, [Y_pred.shape[0]])

            labels = tf.cast(labels, tf.int64)
            Y_pred = tf.cast(Y_pred, tf.int64)

            accuracy = tf.math.equal(labels, Y_pred)
            accuracy = tf.math.reduce_mean(tf.cast(accuracy, tf.float32))
            accuracy = tf.expand_dims(accuracy, axis = 0)

            # generate signals from given seed
            predictions = None
            predictions_raw = None

            for i in range(self.generate_count):
                raw = self.gptGenerator.generator(tf.expand_dims(self.seed[i], axis = 0), training = False)

                if predictions_raw == None:
                    predictions_raw = raw
                else:
                    predictions_raw = tf.concat([predictions_raw, raw], axis = 0)

            for i in range(self.config.class_count):
                avg_prediction = tf.reduce_mean(predictions_raw[i * self.sub_vector_size : (i + 1) * self.sub_vector_size], axis = 0)
                avg_prediction = tf.expand_dims(avg_prediction, axis = 0)

                if predictions is None:
                    predictions = avg_prediction
                else:
                    predictions = tf.concat([predictions, avg_prediction], axis = 0)

            # check whether the training class count equals to average predictions count
            assert self.real_data.shape[0] == predictions.shape[0]

            ## raw signal kl
            kl_losses = None

            for i in range(int(self.real_data.shape[0])):
                real = tf.reshape(self.real_data[i], shape = [1, 2089, 500])
                kl = self.loss_kl(real, predictions[i])
                kl = tf.expand_dims(kl, axis = 0)

                if kl_losses is None:
                    kl_losses = kl
                else:
                    kl_losses = tf.concat([kl_losses, kl], axis = 0)

            ## channels kl
            predictions = tf.reshape(predictions, predictions.shape + [1])
            sigs = tf.concat([predictions, self.real_data], axis = 0)
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
            signal_kl_losses = None
            
            for chan_i in range(int(signals.shape[1])):
                for class_i in range(int(self.config.class_count)):
                    kl = self.loss_kl(signals[class_i, chan_i, :], signals[class_i + int(self.config.class_count), chan_i, :])
                    kl = tf.expand_dims(kl, axis = 0)

                    if signal_kl_losses is None:
                        signal_kl_losses = kl
                    else:
                        signal_kl_losses = tf.concat([signal_kl_losses, kl], axis = 0)

            original_loss = loss
            sum_kl = tf.reduce_sum(kl_losses)
            delta_kl_loss = tf.math.multiply(tf.constant(-self.delta), sum_kl)
            loss = loss + delta_kl_loss

        # update gradient
        grads = tape.gradient(original_loss, self.gptGenerator.generator.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.gptGenerator.generator.trainable_weights))

        result = {}
        result["loss"] = loss
        result["accuracy"] = accuracy
        result["generated"] = predictions
        result["original_loss"] = original_loss
        result["delta_kl_loss"] = delta_kl_loss

        ## record raw signal loss
        for class_i in range(int(self.config.class_count)):
            name = "class{}_signal_kl".format(class_i)
            result[name] = kl_losses[class_i]

        ## record channels signal loss
        channels = ["C3", "C4", "Cz"]
        for chan_i in range(int(signals.shape[1])):
            for class_i in range(int(self.config.class_count)):
                name = "class{}_channel_{}_signal_kl".format(class_i, channels[chan_i])
                result[name] = signal_kl_losses[class_i * int(self.config.class_count) + chan_i]

        X = None
        sigs = None
        brain = None
        signals = None
        kl_losses = None
        signal_kl_losses = None

        del X
        del sigs
        del brain
        del signals
        del kl_losses
        del signal_kl_losses

        return result

    def call(self, inputs):

        predictions = None

        for i in range(inputs.shape[0]):
            prediction = np.asarray(self.gptGenerator.generator(tf.expand_dims(inputs[i], axis = 0), training = False))

            if predictions is None:
                predictions = prediction
            else:
                predictions = tf.concat([predictions, prediction], axis = 0)

        batch = predictions.shape[0]
        filtered_activations = None

        for idx in range(batch):
            (l_act, r_act) = restore_brain_activation(predictions[idx], self.boolean_l, self.boolean_r)
            act = np.concatenate([l_act, r_act], axis = 0)
            eeg_signal = np.dot(self.t_matrix, act)
            eeg_signal = np.expand_dims(eeg_signal, axis = 0)
            motor_signal = None

            for name, i in enumerate(eeg_signal):
                if name in Brain.get_channel_names():
                    sig = np.expand_dims(eeg_signal[i], axis = 0)

                    if motor_signal is None:
                        motor_signal = sig
                    else:
                        motor_signal = np.concatenate([motor_signal, sig], axis = 0)

            if filtered_activations is None:
                filtered_activations = motor_signal
            else:
                filtered_activations = np.concatenate([filtered_activations, motor_signal], axis = 0)

        print("filtered activation shape: {}".format(filtered_activations.shape))
            
        return filtered_activations