
# import random
import numpy as np
import tensorflow as tf
# from tensorflow.python.ops.gen_math_ops import real

from .gpt2 import TFGPT2MainLayer
from .discriminator import Discriminator

from utils.brain_activation import boolean_brain, restore_brain_activation, transformation_matrix, restore_brain_activation_tf

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

        generated_count = 40

        # add one hot vector for every seed
        self.seed = tf.random.normal([generated_count, config.n_positions, config.n_embd])

        tmp = [0] * int(generated_count / 2) + [1] * int(generated_count / 2)
        l = tf.constant(tmp)

        # l = tf.constant([x % 2 for x in range(2)])
        one_hot = tf.one_hot(l, depth = 2)
        one_hot = tf.expand_dims(one_hot, axis = 1)
        one_hot = tf.repeat(one_hot, repeats = config.n_positions, axis = 1)
        self.seed = tf.concat([self.seed, one_hot], axis = 2)

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

    # def build(self, input_shape):
    #     return super().build(input_shape)

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
        
        num_classes = real_labels.shape[-1]

        real_images = self.generate_original_full_brain_activation(real)
        image_size_h = real_images.shape[1]
        image_size_w = real_images.shape[2]

        # one hot information
        one_hot_labels = tf.expand_dims(real_labels, axis = 1)
        one_hot_labels = tf.repeat(one_hot_labels, repeats = self.noise_len, axis = 1)

        # print("one_hot_labels: {}".format(one_hot_labels.shape))

        image_one_hot_labels = real_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats = [image_size_h * image_size_w]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, image_size_h, image_size_w, num_classes)
        )
        
        # Sample random points in the latent space
        batch_size = 8 #real_images.shape[0]
        d_loss = 0

        for _ in range(self.d_extra_steps):

            random_latent_vectors = tf.random.normal(shape = (batch_size, self.noise_len, self.noise_dim))
            random_latent_vectors = tf.concat([random_latent_vectors, one_hot_labels], axis = 2)

            # generate images from gpt2 
            generated_images = self.generator(random_latent_vectors)

            random_latent_vectors = None
            del random_latent_vectors

            # Combine them with real images
            fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
            real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
            combined_images = tf.concat([fake_image_and_labels, real_image_and_labels], axis = 0)
            
            # Assemble labels discriminating real from fake images
            labels = tf.concat(
                [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
            )

            # Add random noise to the labels - important trick!
            labels += 0.05 * tf.random.uniform(tf.shape(labels))

            # Train the disciminator
            with tf.GradientTape() as tape:
                fake_predictions = self.discriminator(fake_image_and_labels)
                real_predictions = self.discriminator(real_image_and_labels)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(real_image_and_labels, fake_image_and_labels, batch_size)
                # Add the gradient penalty to the original discriminator loss
                # d_loss = self.loss_fn(labels, predictions) + gp * self.gp_weight
                d_loss = tf.reduce_mean(fake_predictions) - tf.reduce_mean(real_predictions) + gp * self.gp_weight

            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape = (batch_size, self.noise_len, self.noise_dim))
        random_latent_vectors = tf.concat([random_latent_vectors, one_hot_labels], axis = 2)
        
        # Assemble labels that say "all real images"
        # misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_latent_vectors)
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_image_and_labels)
            # g_loss = self.loss_fn(misleading_labels, predictions)
            g_loss = -1.0 * tf.reduce_mean(predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        random_latent_vectors = None
        one_hot_labels = None
        image_one_hot_labels = None

        del random_latent_vectors
        del one_hot_labels
        del image_one_hot_labels

        # generate image from given seed
        # predictions = self.generator(self.seed, training = False)
        predictions_left = None
        predictions_right = None
        generate_count = 10
        generate_round = 2

        for i in range(generate_round):
            predictions_l = self.generator(self.seed[i * generate_count : (i + 1) * generate_count], training = False)
            predictions_r = self.generator(self.seed[(i + generate_round) * generate_count : (i + generate_round + 1) * generate_count], training = False)

            if predictions_left == None:
                predictions_left = predictions_l
            else:
                predictions_left = tf.concat([predictions_left, predictions_l], axis = 0)

            if predictions_right == None:
                predictions_right = predictions_r
            else:
                predictions_right = tf.concat([predictions_right, predictions_r], axis = 0)

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
        
        return {"d_loss": d_loss, "g_loss": g_loss, 
                "raw_feet_signal_kl": raw_signal_feet_kl, 
                "raw_tongue_signal_kl": raw_signal_tongue_kl, 
                "raw_feet_spectrum_c3_signal_kl": raw_feet_spectrum_c3_signal_kl, 
                "raw_tongue_spectrum_c3_signal_kl": raw_tongue_spectrum_c3_signal_kl, 
                "raw_feet_spectrum_c4_signal_kl": raw_feet_spectrum_c4_signal_kl, 
                "raw_tongue_spectrum_c4_signal_kl": raw_tongue_spectrum_c4_signal_kl, 
                "raw_feet_spectrum_cz_signal_kl": raw_feet_spectrum_cz_signal_kl, 
                "raw_tongue_spectrum_cz_signal_kl": raw_tongue_spectrum_cz_signal_kl, 
                "generated": predictions}

    def call(self, inputs):
        
        predictions = np.asarray(self.generator(inputs, training = True))

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