
from datetime import time
import numpy as np
import tensorflow as tf

from .model_monitor import generate_and_save_images, save_result_as_gif, save_distribution_record
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
        
        current = logs.get("g_loss")
        d = logs.get("d_loss")

        print("d_loss: {}, g_loss: {}".format(d, current))

        tf.summary.scalar('g_loss', current, step = epoch)
        tf.summary.scalar('d_loss', d, step = epoch)

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


class RecordGeneratedImages(tf.keras.callbacks.Callback):

    def __init__(self, time, n_round, model_name, close_data, open_data):
        super(RecordGeneratedImages, self).__init__()
        self.time = time
        self.n_round = n_round
        self.model_name = model_name

        self.close_data = np.asarray(close_data)
        self.open_data = np.asarray(open_data)
        self.close_data = np.reshape(self.close_data, [2089, 500])
        self.open_data = np.reshape(self.open_data, [2089, 500])

        (self.boolean_l, self.boolean_r) = boolean_brain()
        self.tranformation_matrix = transformation_matrix()
        self.brain_template = fetch_brain_template()

    def on_epoch_end(self, epoch, logs = None):

        generated = logs.get("generated")

        event_count = 8

        for idx in range(event_count):

            tmp_fake = generated

            close_event_idx = idx
            open_event_idx = idx + event_count

            eye_close_event = tmp_fake[close_event_idx]
            eye_open_event = tmp_fake[open_event_idx]

            eye_close_event = np.reshape(eye_close_event, [2089, 500])
            eye_open_event = np.reshape(eye_open_event, [2089, 500])
            
            # generate_and_save_images(   predictions = predictions, 
            #                             time = self.time, 
            #                             n_round =  self.n_round, 
            #                             epoch = epoch, 
            #                             model_name = self.model_name    )


            # save_result_as_gif( time = self.time, 
            #                     model_name = self.model_name, 
            #                     n_round = self.n_round  )

            distribution = [self.close_data, self.open_data, eye_close_event, eye_open_event]
            save_distribution_record(   data = distribution, 
                                        epoch = epoch, 
                                        time = self.time, 
                                        model_name = self.model_name, 
                                        n_round = self.n_round, 
                                        event_idx = idx  )

            (left_brain_eye_close_activation, right_brain_eye_close_activation) = restore_brain_activation(eye_close_event, self.boolean_l, self.boolean_r)
            (left_real_eye_close_activation, right_real_eye_close_activation) = restore_brain_activation(self.close_data, self.boolean_l, self.boolean_r)
            real_close = np.concatenate([left_real_eye_close_activation, right_real_eye_close_activation], axis = 0)

            (left_brain_eye_open_activation, right_brain_eye_open_activation) = restore_brain_activation(eye_open_event, self.boolean_l, self.boolean_r)
            (left_real_eye_open_activation, right_real_eye_open_activation) = restore_brain_activation(self.open_data, self.boolean_l, self.boolean_r)
            real_open = np.concatenate([left_real_eye_open_activation, right_real_eye_open_activation], axis = 0)


            # generate_eeg(real, left_brain_activation, right_brain_activation, self.tranformation_matrix, epoch, self.time, self.model_name, self.n_round)
            generate_single_channel_eeg_signal(real_close, real_open, left_brain_eye_close_activation, right_brain_eye_close_activation, left_brain_eye_open_activation, right_brain_eye_open_activation, self.tranformation_matrix, epoch, self.time, self.model_name, self.n_round, idx)
            generate_mne_plot(self.brain_template, real_close, real_open, left_brain_eye_close_activation, right_brain_eye_close_activation, left_brain_eye_open_activation, right_brain_eye_open_activation, self.tranformation_matrix, epoch, self.time, self.model_name, self.n_round, idx)

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