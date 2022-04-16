
import random
import numpy as np
import tensorflow as tf

from res.brain import Brain
from .gpt2 import TFGPT2MainLayer
from .discriminator import Discriminator
from utils.brain_activation import boolean_brain, restore_brain_activation, transformation_matrix, restore_brain_activation_tf
from utils.datasetGenerator import generate_random_vectors

SUBGROUP_SIZE = 1
CHANNEL_NAME  = [   "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "C3", 
                    "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6", "AFz", "P7", "P3", "Pz", "P4",
                    "P8", "PO9", "O1", "Oz", "O2", "PO10", "AF7", "AF3", "AF4", "AF8", "F5", "F1",
                    "F2", "F6", "FT7", "FC3", "FC4", "FT8", "C1", "C2", "C6", "TP7", "CP3", "CPz",
                    "CP4", "TP8", "P5", "P1", "P2", "P6", "PO7", "PO3", "POz", "PO4", "PO8" ]

class gpt2cgan(tf.keras.Model):

    def __init__(self, config, data_avg, d_extra_steps: int = 5, last_dim: int = 1, **kwargs):
        super(gpt2cgan, self).__init__()

        self.generator = TFGPT2MainLayer(config = config, name = "name", last_dim = last_dim)
        self.discriminator = Discriminator(config = config)
        self.noise_len = config.n_positions
        self.noise_dim = config.n_embd

        self.gp_weight = 10.0
        self.d_extra_steps = d_extra_steps

        self.generate_count = config.example_to_generate
        assert self.generate_count % config.class_count == 0

        # add one hot vector for every seed
        self.seed, _, self.sub_vector_size = generate_random_vectors(self.generate_count, config.n_positions, config.n_embd, config.class_rate_random_vector, config.class_count, config.noise_variance, False)
        self.last_dim = last_dim

        self.config = config

        # self.t_matrix = np.asarray(transformation_matrix())
        self.t_matrix = tf.constant(transformation_matrix(), dtype = tf.float32)
        (self.boolean_l, self.boolean_r) = boolean_brain()

        self.real_data = tf.constant(data_avg, dtype = tf.float32)
        self.real_data = tf.reshape(self.real_data, shape = [self.real_data.shape[0], self.real_data.shape[1], self.real_data.shape[2], 1])

    def compile(self, d_optimizer, g_optimizer, loss_fn, loss_kl):
        super(gpt2cgan, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.loss_kl = loss_kl

    def gradient_penalty(self, 
                         real_images, 
                         fake_images, 
                         batch_size: int = 8):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training = True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]

        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis = [1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def generate_original_full_brain_activation(self, original_images):

        train_data = tf.constant([])

        left_data = original_images[:, :1022, :]
        right_data = original_images[:, 1022:, :]

        left_shape = left_data.shape
        right_shape = right_data.shape

        epoch = 8
        timestamp = 500

        left_vertex_count = int(left_shape[1] / SUBGROUP_SIZE)
        right_vertex_count = int(right_shape[1] / SUBGROUP_SIZE)

        left_data = left_data[:epoch, :left_vertex_count * SUBGROUP_SIZE, :timestamp]   
        right_data = right_data[:epoch, :right_vertex_count * SUBGROUP_SIZE, :timestamp]

        concat_data = tf.concat([left_data, right_data], axis = 1)

        start = False

        for i in range(int(concat_data.shape[1] / SUBGROUP_SIZE)):
            if not start:
                train_data = concat_data[:epoch, SUBGROUP_SIZE * i:(SUBGROUP_SIZE * i) + 1, :timestamp]
                start = True
            else:
                train_data = tf.concat((train_data, concat_data[:epoch, SUBGROUP_SIZE * i:(SUBGROUP_SIZE * i) + 1, :timestamp]), axis = 1)

        return train_data


    def train_step(self, data):

        real, real_labels = data
        real_images = self.generate_original_full_brain_activation(real)
        
        # Sample random points in the latent space
        batch_size = self.config.batch_size
        d_loss = 0

        for _ in range(self.d_extra_steps):
            (random_latent_vectors, _, _) = generate_random_vectors(self.generate_count, self.config.n_positions, self.config.n_embd, self.config.class_rate_random_vector, self.config.class_count, self.config.noise_variance, False)
            random_latent_vectors = tf.constant(random_latent_vectors.tolist(), dtype = tf.float32)
            
            # generate images from gpt2 
            generated_images = self.generator(random_latent_vectors)

            random_latent_vectors = None
            del random_latent_vectors

            # Combine them with real images
            fake_image_and_labels = generated_images# tf.concat([generated_images, image_one_hot_labels], -1)
            real_image_and_labels = real_images #tf.concat([real_images, image_one_hot_labels], -1)
            fake_image_and_labels = tf.expand_dims(fake_image_and_labels, axis = 3)
            
            # Assemble labels discriminating real from fake images
            labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis = 0)

            # Add random noise to the labels - important trick!
            labels += 0.05 * tf.random.uniform(tf.shape(labels))

            # Train the disciminator
            with tf.GradientTape() as tape:
                fake_predictions = self.discriminator(fake_image_and_labels)
                real_predictions = self.discriminator(real_image_and_labels)
                
                # Calculate the gradient penalty
                gp = self.gradient_penalty(real_image_and_labels, fake_image_and_labels, batch_size)

                # Add the gradient penalty to the original discriminator loss
                d_loss = tf.reduce_mean(fake_predictions) - tf.reduce_mean(real_predictions) + gp * self.gp_weight

            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Sample random points in the latent space
        (random_latent_vectors, _, _) = generate_random_vectors(self.generate_count, self.config.n_positions, self.config.n_embd, self.config.class_rate_random_vector, self.config.class_count, self.config.noise_variance, False)
        random_latent_vectors = tf.constant(random_latent_vectors.tolist(), dtype = tf.float32)

        # Train the generator (note that we should *not* update the weights of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_latent_vectors)
            fake_images = tf.expand_dims(fake_images, axis = 3)
            predictions = self.discriminator(fake_images)
            g_loss = -1.0 * tf.reduce_mean(predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        random_latent_vectors = None
        one_hot_labels = None

        del random_latent_vectors
        del one_hot_labels

        # generate signals from given seed
        predictions = None
        predictions_raw = None

        for i in range(self.generate_count):
            raw = self.generator(tf.expand_dims(self.seed[i], axis = 0), training = False)

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
                kl = self.loss_kl(signals[2, chan_i, :], signals[class_i, 0, :])
                kl = tf.expand_dims(kl, axis = 0)

                if signal_kl_losses is None:
                    signal_kl_losses = kl
                else:
                    signal_kl_losses = tf.concat([signal_kl_losses, kl], axis = 0)

        result = {}
        result["d_loss"] = d_loss
        result["g_loss"] = g_loss
        result["generated"] = predictions

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

        sigs = None
        brain = None
        signals = None
        kl_losses = None
        signal_kl_losses = None

        del sigs
        del brain
        del signals
        del kl_losses
        del signal_kl_losses

        return result
        
        # return {"d_loss": d_loss, "g_loss": g_loss, 
        #         "raw_feet_signal_kl": raw_signal_feet_kl, 
        #         "raw_tongue_signal_kl": raw_signal_tongue_kl, 
        #         "raw_feet_spectrum_c3_signal_kl": raw_feet_spectrum_c3_signal_kl, 
        #         "raw_tongue_spectrum_c3_signal_kl": raw_tongue_spectrum_c3_signal_kl, 
        #         "raw_feet_spectrum_c4_signal_kl": raw_feet_spectrum_c4_signal_kl, 
        #         "raw_tongue_spectrum_c4_signal_kl": raw_tongue_spectrum_c4_signal_kl, 
        #         "raw_feet_spectrum_cz_signal_kl": raw_feet_spectrum_cz_signal_kl, 
        #         "raw_tongue_spectrum_cz_signal_kl": raw_tongue_spectrum_cz_signal_kl, 
        #         "generated": predictions}

    def call(self, inputs):

        predictions = None

        for i in range(inputs.shape[0]):
            prediction = np.asarray(self.generator(tf.expand_dims(inputs[i], axis = 0), training = False))

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