
from datetime import time
import numpy as np
import tensorflow as tf

from .model_monitor import generate_and_save_images, save_result_as_gif, save_distribution_record, record_model_weight
from .brain_activation import boolean_brain, transformation_matrix, restore_brain_activation, generate_eeg, generate_single_channel_eeg_signal, fetch_brain_template, generate_mne_plot

class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

    Arguments:
        patience: Number of epochs to wait after min has been hit. After this
        number of no improvement, training stops.
    """

    def __init__(self, patience = 5):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs = None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0

        # The epoch the training stops at.
        self.stopped_epoch = 0

        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs = None):
        
        current = logs.get("loss")
        # d = logs.get("d_loss")

        # print("d_loss: {}, g_loss: {}".format(d, current))

        tf.summary.scalar('g_loss', current, step = epoch)
        # tf.summary.scalar('d_loss', d, step = epoch)

        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience and current < 0.1:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs = None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

class RecordWeight(tf.keras.callbacks.Callback):
    
    def __init__(self):
        super(RecordWeight, self).__init__()

    def on_epoch_end(self, epoch, logs = None):
        weight = logs.get("weight")
        record_model_weight(weight.h[0].attn.get_weights())


class RecordGeneratedImages(tf.keras.callbacks.Callback):

    def __init__(self, time, n_round, model_name, close_data, open_data, raw_close, raw_open):
        super(RecordGeneratedImages, self).__init__()
        self.time = time
        self.n_round = n_round
        self.model_name = model_name

        close_len = len(close_data)
        open_len = len(open_data)
        self.close_data = np.asarray(close_data)
        self.open_data = np.asarray(open_data)
        self.close_data = np.reshape(self.close_data, [close_len, 2089, 500])
        self.open_data = np.reshape(self.open_data, [open_len, 2089, 500])

        self.raw_close = raw_close
        self.raw_open = raw_open

        (self.boolean_l, self.boolean_r) = boolean_brain()
        self.tranformation_matrix = transformation_matrix()
        self.brain_template = fetch_brain_template()

        self.real_close = np.asarray([])
        self.real_open = np.asarray([])

        for idx in range(close_len):
            (left_real_eye_close_activation, right_real_eye_close_activation) = restore_brain_activation(self.close_data[idx], self.boolean_l, self.boolean_r)
            tmp = np.concatenate([left_real_eye_close_activation, right_real_eye_close_activation], axis = 0)
            tmp = np.expand_dims(tmp, axis = 0)

            if len(self.real_close) == 0:
                self.real_close = tmp
            else:
                self.real_close = np.concatenate([self.real_close, tmp], axis = 0)

        for idx in range(open_len):
            (left_real_eye_open_activation, right_real_eye_open_activation) = restore_brain_activation(self.open_data[idx], self.boolean_l, self.boolean_r)
            tmp = np.concatenate([left_real_eye_open_activation, right_real_eye_open_activation], axis = 0)

            tmp = np.expand_dims(tmp, axis = 0)

            if len(self.real_open) == 0:
                self.real_open = tmp
            else:
                self.real_open = np.concatenate([self.real_open, tmp], axis = 0)

        print("-" * 100)
        print("RecordGeneratedImages:")
        print("real_close shape: {}, real_open shape: {}".format(self.real_close.shape, self.real_open.shape))


    def on_epoch_end(self, epoch, logs = None):

        if (epoch + 1) % 50 == 0 || epoch == 0:

            generated = logs.get("generated")

            eye_close_event = np.reshape(generated[:80], [80, 2089, 500])
            eye_open_event = np.reshape(generated[80:], [80, 2089, 500])
            
            # generate_and_save_images(   predictions = predictions, 
            #                             time = self.time, 
            #                             n_round =  self.n_round, 
            #                             epoch = epoch, 
            #                             model_name = self.model_name    )


            # save_result_as_gif( time = self.time, 
            #                     model_name = self.model_name, 
            #                     n_round = self.n_round  )

            distribution = [self.close_data[0], self.open_data[0], eye_close_event[0], eye_open_event[0]]
            save_distribution_record(   data = distribution, 
                                        epoch = epoch, 
                                        time = self.time, 
                                        model_name = self.model_name, 
                                        n_round = self.n_round  )

            left_brain_eye_close_activation = np.asarray([])
            right_brain_eye_close_activation = np.asarray([])
            left_brain_eye_open_activation = np.asarray([])
            right_brain_eye_open_activation = np.asarray([])

            generate_count = 80

            if (epoch + 1) == 1000:
                generate_count = 1

                eye_close_event = tf.reduce_mean(eye_close_event, axis = 0)
                eye_open_event = tf.reduce_mean(eye_open_event, axis = 0)

                print("-" * 100)
                print("eye_close_event shape: {}".format(eye_close_event.shape))
                print("eye_open_event shape: {}".format(eye_open_event.shape))
                print("-" * 100)

            for idx in range(generate_count):
                (l_close_tmp, r_close_tmp) = restore_brain_activation(eye_close_event[idx], self.boolean_l, self.boolean_r)
                l_close_tmp = np.expand_dims(l_close_tmp, axis = 0)
                r_close_tmp = np.expand_dims(r_close_tmp, axis = 0)

                (l_open_tmp, r_open_tmp) = restore_brain_activation(eye_open_event[idx], self.boolean_l, self.boolean_r)
                l_open_tmp = np.expand_dims(l_open_tmp, axis = 0)
                r_open_tmp = np.expand_dims(r_open_tmp, axis = 0)

                if len(left_brain_eye_close_activation) == 0:
                    left_brain_eye_close_activation = l_close_tmp
                else:
                    left_brain_eye_close_activation = np.concatenate([left_brain_eye_close_activation, l_close_tmp], axis = 0)

                if len(right_brain_eye_close_activation) == 0:
                    right_brain_eye_close_activation = r_close_tmp
                else:
                    right_brain_eye_close_activation = np.concatenate([right_brain_eye_close_activation, r_close_tmp], axis = 0)

                if len(left_brain_eye_open_activation) == 0:
                    left_brain_eye_open_activation = l_open_tmp
                else:
                    left_brain_eye_open_activation = np.concatenate([left_brain_eye_open_activation, l_open_tmp], axis = 0)

                if len(right_brain_eye_open_activation) == 0:
                    right_brain_eye_open_activation = r_open_tmp
                else:
                    right_brain_eye_open_activation = np.concatenate([right_brain_eye_open_activation, r_open_tmp], axis = 0)


            # generate_eeg(real, left_brain_activation, right_brain_activation, self.tranformation_matrix, epoch, self.time, self.model_name, self.n_round)
            generate_single_channel_eeg_signal(self.raw_close, self.open_data, self.real_close, self.real_open, left_brain_eye_close_activation, right_brain_eye_close_activation, left_brain_eye_open_activation, right_brain_eye_open_activation, self.tranformation_matrix, epoch, self.time, self.model_name, self.n_round, idx)
            generate_mne_plot(self.brain_template, self.real_close, self.real_open, left_brain_eye_close_activation, right_brain_eye_close_activation, left_brain_eye_open_activation, right_brain_eye_open_activation, self.tranformation_matrix, epoch, self.time, self.model_name, self.n_round, idx)

            tmp_fake = None
            eye_close_event = None
            eye_open_event = None
            left_brain_eye_close_activation = None
            right_brain_eye_close_activation = None
            left_brain_eye_open_activation = None
            right_brain_eye_open_activation = None
            distribution = None

            del tmp_fake
            del eye_close_event
            del eye_open_event
            del left_brain_eye_close_activation
            del right_brain_eye_close_activation
            del left_brain_eye_open_activation
            del right_brain_eye_open_activation
            del distribution

            del generated