
from datetime import time
import numpy as np
import tensorflow as tf

from .model_monitor import generate_and_save_images, save_result_as_gif, save_distribution_record

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

    def __init__(self, time, n_round, model_name):
        super(RecordGeneratedImages, self).__init__()
        self.time = time
        self.n_round = n_round
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs = None):
        # predictions = logs.get("predictions")
        distribution = logs.get("distribution")

        # generate_and_save_images(   predictions = predictions, 
        #                             time = self.time, 
        #                             n_round =  self.n_round, 
        #                             epoch = epoch, 
        #                             model_name = self.model_name    )


        # save_result_as_gif( time = self.time, 
        #                     model_name = self.model_name, 
        #                     n_round = self.n_round  )

        save_distribution_record(   data = distribution, 
                                    epoch = epoch, 
                                    time = self.time, 
                                    model_name = self.model_name, 
                                    n_round = self.n_round  )
                            