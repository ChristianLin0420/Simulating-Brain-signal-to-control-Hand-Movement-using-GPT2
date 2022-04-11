
from calendar import c
import os
import io
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from .model_monitor import record_model_weight
from .brain_activation import boolean_brain, transformation_matrix, restore_brain_activation, fetch_brain_template, restore_brain_activation_tf

'''
----- Accuracy -----
@desciption:
    ...
'''
class Accuracy(tf.keras.callbacks.Callback):

    def __init__(self, config, time, _round):
        self.config = config
        self.time = time
        self.round = _round

        self.batch_accuracy = []
        self.accuracy = []

    def on_train_batch_end(self, epoch, logs = None):
        self.batch_accuracy.append(float(logs.get("accuracy")))

    def on_epoch_end(self, epoch, logs = None):
        mean_accuracy = float(sum(self.batch_accuracy)) / float(len(self.batch_accuracy))
        self.accuracy.append(mean_accuracy)
        self.batch_accuracy = list()

        data = { "accuracy" : self.accuracy }

        if epoch == self.config.epochs - 1:
            with io.open("result/{}/{}/history/{}/{}.json".format(self.config.model_name, self.time, self.round, "accuracy"), 'w', encoding = 'utf8') as outfile:
                s = json.dumps(data, indent = 4, sort_keys = True, ensure_ascii = False)
                outfile.write(s)
            
            self.batch_accuracy = None
            self.accuracy = None

'''
----- Loss -----
@desciption:
    ...
'''
class Loss(tf.keras.callbacks.Callback):

    def __init__(self, config, time, _round):
        self.config = config
        self.time = time
        self.round = _round

        self.batch_loss = []
        self.loss = []

    def on_train_batch_end(self, epoch, logs = None):
        self.batch_loss.append(float(logs.get("loss")))

    def on_epoch_end(self, epoch, logs = None):
        mean_loss = float(sum(self.batch_loss)) / float(len(self.batch_loss))
        self.loss.append(mean_loss)
        self.batch_loss = list()

        data = { "loss" : self.loss}

        if epoch == self.config.epochs - 1:
            with io.open("result/{}/{}/history/{}/{}.json".format(self.config.model_name, self.time, self.round, "loss"), 'w', encoding = 'utf8') as outfile:
                s = json.dumps(data, indent = 4, sort_keys = True, ensure_ascii = False)
                outfile.write(s)
            
            self.batch_loss = None
            self.loss = None

'''
----- Loss -----
@desciption:
    ...
'''
class GANLoss(tf.keras.callbacks.Callback):

    def __init__(self, config, time, _round):
        self.config = config
        self.time = time
        self.round = _round

        self.batch_d_loss = list()
        self.batch_g_loss = list()
        self.d_loss = list()
        self.g_loss = list()

    def on_train_batch_end(self, epoch, logs = None):
        self.batch_d_loss.append(float(logs.get('d_loss')))
        self.batch_g_loss.append(float(logs.get('g_loss')))

    def on_epoch_end(self, epoch, logs = None):
        mean_d_loss = float(sum(self.batch_d_loss)) / float(len(self.batch_d_loss))
        mean_g_loss = float(sum(self.batch_g_loss)) / float(len(self.batch_g_loss))
        self.d_loss.append(mean_d_loss)
        self.g_loss.append(mean_g_loss)
        self.batch_d_loss = list()
        self.batch_g_loss = list()

        d_data = { "d_loss" : self.d_loss}
        g_data = { "g_loss" : self.g_loss}

        if epoch == self.config.epochs - 1:
            with io.open("result/{}/{}/history/{}/{}.json".format(self.config.model_name, self.time, self.round, "d_loss"), 'w', encoding = 'utf8') as outfile:
                s = json.dumps(d_data, indent = 4, sort_keys = True, ensure_ascii = False)
                outfile.write(s)

            with io.open("result/{}/{}/history/{}/{}.json".format(self.config.model_name, self.time, self.round, "g_loss"), 'w', encoding = 'utf8') as outfile:
                s = json.dumps(g_data, indent = 4, sort_keys = True, ensure_ascii = False)
                outfile.write(s)
            
            self.batch_d_loss = None
            self.batch_g_loss = None
            self.d_loss = None
            self.g_loss = None

'''
----- STFTgenerator -----
@desciption:
    ...
'''
class STFTgenerator(tf.keras.callbacks.Callback):

    def __init__(self, config, time, _round):
        self.config = config
        self.time = time
        self.round = _round

        self.stft = None
        (self.boolean_l, self.boolean_r) = boolean_brain()
        self.transformation_matrix = transformation_matrix()

    def on_train_batch_end(self, epoch, logs = None):
        if epoch % 10 == 0:
            self.stft = tf.constant(logs.get('generated'))

    def on_epoch_end(self, epoch, logs = None):
        if epoch % 10 == 0:
            # data = np.asarray(logs.get('generated')).tolist()
            # if not os.path.exists('result/{}/{}/generated/{}/epoch_{:04d}'.format(self.config.model_name, self.time, self.round, epoch)):
            #     os.mkdir('result/{}/{}/generated/{}/epoch_{:04d}'.format(self.config.model_name, self.time, self.round, epoch))
            
            # data = { "generated" : data }
            # with io.open("result/{}/{}/generated/{}/epoch_{:04d}/{}.json".format(self.config.model_name, self.time, self.round, int(epoch), "generated"), 'w', encoding = 'utf8') as outfile:
            #     s = json.dumps(data, indent = 4, sort_keys = True, ensure_ascii = False)
            #     outfile.write(s)
            
            self.stft = self.stft / self.stft.shape[0]
            brain = None
            signals = None

            ## reconstruct the brain voxels from the given generated output
            for i in range(self.stft.shape[0]):
                (l_tmp, r_tmp) = restore_brain_activation_tf(self.stft[i], self.boolean_l, self.boolean_r)
                brain_activation = tf.concat([l_tmp, r_tmp], axis = 0)
                brain_activation = tf.expand_dims(brain_activation, axis = 0)

                if brain == None:
                    brain = brain_activation
                else:
                    brain = tf.concat([brain, brain_activation], axis = 0)

            ## calculate the average value of the generated signals
            tmp = tf.reshape(brain, shape = [brain.shape[0], brain.shape[1], brain.shape[2]])
            brain = None

            generated_num_per_class = int(self.config.example_to_generate / self.config.class_count)
            
            for i in range(self.config.class_count):
                mean = tf.reduce_mean(tmp[i * generated_num_per_class : (i + 1) * generated_num_per_class], axis = 0)
                mean = tf.expand_dims(mean, axis = 0)

                if brain is None:
                    brain = mean
                else:
                    brain = tf.concat([brain, mean], axis = 0)
                

            ## forward implenmentation (brain voxels -> EEG signals)
            for i in range(brain.shape[0]):
                signal = tf.matmul(self.transformation_matrix, brain[i])
                signal = tf.expand_dims(signal, axis = 0)

                if signals == None:
                    signals = signal
                else:
                    signals = tf.concat([signals, signal], axis = 0)

            signals = signals[:, 11:14, :]

            Zxx = tf.signal.stft(signals, frame_length = 256, frame_step = 16)
            Zxx = tf.abs(Zxx)
            
            bz = int(self.stft.shape[0])
            channels = ["C3", "C4", "Cz"]

            ## generate short-time fourier transform figures
            for sample in range(bz):
                if not os.path.exists('result/{}/{}/stft/{}/epoch_{:04d}'.format(self.config.model_name, self.time, self.round, epoch)):
                    os.mkdir('result/{}/{}/stft/{}/epoch_{:04d}'.format(self.config.model_name, self.time, self.round, epoch))

                for idx in range(int(len(channels))):
                    log_spec = tf.math.log(tf.transpose(Zxx[sample][idx]))
                    height = 40
                    width = log_spec.shape[1]
                    x_axis = tf.linspace(0, 2, num = width)
                    y_axis = range(height)
                    plt.pcolormesh(x_axis, y_axis, log_spec[:40, ])
                    plt.title('STFT Magnitude for channel {} of class {} in iteration {}'.format(channels[idx], sample + 1, epoch))
                    plt.ylabel('Frequency [Hz]')
                    plt.xlabel('Time [sec]')
                    plt.savefig('result/{}/{}/stft/{}/epoch_{:04d}/class_{}_{}.png'.format(self.config.model_name, self.time, self.round, epoch, sample + 1, channels[idx]))
                    plt.close()
























class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

    Arguments:
        patience: Number of epochs to wait after min has been hit. After this
        number of no improvement, training stops.
    """

    def __init__(self, patience = 10):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        self.epsilon = 0.01
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
        
        current = float(logs.get("d_loss"))
        # d = logs.get("d_loss")

        # print("d_loss: {}, g_loss: {}".format(d, current))

        tf.summary.scalar('d_loss', current, step = epoch)
        # tf.summary.scalar('d_loss', d, step = epoch)

        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience and current <= 0.5 + self.epsilon and current >= 0.5 - self.epsilon:
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


        if (epoch + 1) % 20 == 0 or epoch == 0:

            generated = logs.get("generated")

            eye_close_event = np.reshape(generated[:1], [1, 2089, 500])
            eye_open_event = np.reshape(generated[1:2], [1, 2089, 500])
            
            # generate_and_save_images(   predictions = predictions, 
            #                             time = self.time, 
            #                             n_round =  self.n_round, 
            #                             epoch = epoch, 
            #                             model_name = self.model_name    )


            # save_result_as_gif( time = self.time, 
            #                     model_name = self.model_name, 
            #                     n_round = self.n_round  )

            # distribution = [self.close_data[0], self.open_data[0], eye_close_event[0], eye_open_event[0]]
            # save_distribution_record(   data = distribution, 
            #                             epoch = epoch, 
            #                             time = self.time, 
            #                             model_name = self.model_name, 
            #                             n_round = self.n_round  )

            left_brain_eye_close_activation = np.asarray([])
            right_brain_eye_close_activation = np.asarray([])
            left_brain_eye_open_activation = np.asarray([])
            right_brain_eye_open_activation = np.asarray([])

            generate_count = 1

            if (epoch + 1) == 1000:
                generate_count = 1

                eye_close_event = tf.reduce_mean(eye_close_event, axis = 0)
                eye_open_event = tf.reduce_mean(eye_open_event, axis = 0)

                eye_close_event = tf.expand_dims(eye_close_event, axis = 0)
                eye_open_event = tf.expand_dims(eye_open_event, axis = 0)

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
            # generate_single_channel_eeg_signal( self.raw_close, self.open_data, 
            #                                     self.real_close, self.real_open, 
            #                                     left_brain_eye_close_activation, 
            #                                     right_brain_eye_close_activation, 
            #                                     left_brain_eye_open_activation, 
            #                                     right_brain_eye_open_activation, 
            #                                     self.tranformation_matrix, 
            #                                     epoch, self.time, self.model_name, self.n_round, idx)

            # generate_mne_plot(  self.brain_template, self.real_close, 
            #                     self.real_open, 
            #                     left_brain_eye_close_activation, 
            #                     right_brain_eye_close_activation, 
            #                     left_brain_eye_open_activation, 
            #                     right_brain_eye_open_activation, 
            #                     self.tranformation_matrix, 
            #                     epoch, self.time, self.model_name, self.n_round, idx)

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

class RecordReconstructedGeneratedImages(tf.keras.callbacks.Callback):

    def __init__(self, time, n_round, model_name, real_data, model, variance: float = 1.0):
        super(RecordReconstructedGeneratedImages, self).__init__()
        self.time = time
        self.n_round = n_round
        self.model_name = model_name
        self.variance = variance

        (self.boolean_l, self.boolean_r) = boolean_brain()
        self.transformation_matrix = transformation_matrix()
        self.brain_template = fetch_brain_template()

        self.t_accuracy_collection = []
        self.t_loss_collection = []
        self.t_original_collection = []
        self.t_kl_delta_loss = []

        self.accuracy_collection = []
        self.loss_collection = []
        self.orignal_loss_collection = []
        self.kl_delta_loss = []

        self.real_feat_data = real_data[0]
        self.real_tongue_data = real_data[1]

        if not os.path.exists("results/img_results/{}/{}/signal".format(model_name, time)):
            os.mkdir("results/img_results/{}/{}/signal".format(model_name, time))

        if not os.path.exists("results/img_results/{}/{}/stft".format(model_name, time)):
            os.mkdir("results/img_results/{}/{}/stft".format(model_name, time))

        if not os.path.exists("results/img_results/{}/{}/json".format(model_name, time)):
            os.mkdir("results/img_results/{}/{}/json".format(model_name, time))

        self.model = model

    def on_train_batch_end(self, batch, logs=None):                
        self.t_accuracy_collection.append(float(logs.get('accuracy')))
        self.t_loss_collection.append(float(logs.get('loss')))
        self.t_original_collection.append(float(logs.get('original_loss')))
        self.t_kl_delta_loss.append(float(logs.get('delta_kl_loss')))

    def on_epoch_end(self, epoch, logs = None):

        acc = 0.0
        loss = 0.0
        o_loss = 0.0
        delta_kl = 0.0

        for a, l in zip(self.t_accuracy_collection, self.t_loss_collection):
            acc += a
            loss += l

        # print("")
        # print("=" * 90)
        # print("t_accuracy_collection: {}".format(self.t_accuracy_collection))
        # print("t_loss_collection: {}".format(self.t_loss_collection))
        # print("before acc: {}, loss: {}".format(acc, loss))

        acc = acc / float(len(self.t_accuracy_collection))
        loss = loss / float(len(self.t_loss_collection))

        # print("after acc: {}, loss: {}".format(acc, loss))
        # print("=" * 90)

        self.accuracy_collection.append(acc)
        self.loss_collection.append(loss)

        self.t_accuracy_collection = []
        self.t_loss_collection = []

        for a, l in zip(self.t_original_collection, self.t_kl_delta_loss):
            o_loss += a
            delta_kl += l

        o_loss = o_loss / float(len(self.t_original_collection))
        delta_kl = delta_kl / float(len(self.t_kl_delta_loss))

        self.orignal_loss_collection.append(o_loss)
        self.kl_delta_loss.append(delta_kl)

        self.t_original_collection = []
        self.t_kl_delta_loss = []

        if epoch == 4999:
            data = {
                "accuracy" : self.accuracy_collection,
                "loss" : self.loss_collection,
                "original_loss": self.orignal_loss_collection,
                "kl_delta_loss": self.kl_delta_loss
            }

            if os.path.exists("results/img_results/{}/{}/json/1_variance_{}.json".format(self.model_name, self.time, self.variance)):
                if os.path.exists("results/img_results/{}/{}/json/2_variance_{}.json".format(self.model_name, self.time, self.variance)):
                    if not os.path.exists("results/img_results/{}/{}/json/3_variance_{}.json".format(self.model_name, self.time, self.variance)):
                        with io.open("results/img_results/{}/{}/json/3_variance_{}.json".format(self.model_name, self.time, self.variance), 'w', encoding='utf8') as outfile:
                            str_ = json.dumps(data, indent=4, sort_keys=True, ensure_ascii=False)
                            outfile.write(str_)

            if os.path.exists("results/img_results/{}/{}/json/1_variance_{}.json".format(self.model_name, self.time, self.variance)):
                if not os.path.exists("results/img_results/{}/{}/json/2_variance_{}.json".format(self.model_name, self.time, self.variance)):
                    with io.open("results/img_results/{}/{}/json/2_variance_{}.json".format(self.model_name, self.time, self.variance), 'w', encoding='utf8') as outfile:
                        str_ = json.dumps(data, indent=4, sort_keys=True, ensure_ascii=False)
                        outfile.write(str_)

            if not os.path.exists("results/img_results/{}/{}/json/1_variance_{}.json".format(self.model_name, self.time, self.variance)):
                with io.open("results/img_results/{}/{}/json/1_variance_{}.json".format(self.model_name, self.time, self.variance), 'w', encoding='utf8') as outfile:
                    str_ = json.dumps(data, indent=4, sort_keys=True, ensure_ascii=False)
                    outfile.write(str_)

        if (epoch + 1) % 10 == 0:
            self.model.delta /= 10
            tf.print("self.model.delta: {}".format(self.model.delta))

        if (epoch + 1) % 10 == 0 or epoch == 0:

            ## update delta 
            # self.model.delta /= 10
            # tf.print("self.model.delta: {}".format(self.model.delta))
            # save_loss_range_record(np.arange(len(self.loss_collection)), self.accuracy_collection, self.time, "gpt2xcnn", "accuracy")

            sigs = tf.constant(logs.get("generated"))

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

            Zxx = tf.signal.stft(signals, frame_length=256, frame_step=16)
            Zxx = tf.abs(Zxx)
            
            bz = int(sigs.shape[0])
            channels = ["C3", "C4", "Cz"]

            for sample in range(bz):

                hand = "feet"

                if sample == 1:
                    hand = "tongue"
                
                if not os.path.exists('results/img_results/{}/{}/stft/{:04d}_{}'.format(self.model_name, self.time, sample, hand)):
                    os.mkdir('results/img_results/{}/{}/stft/{:04d}_{}'.format(self.model_name, self.time, sample, hand))
                    os.mkdir('results/img_results/{}/{}/signal/{:04d}_{}'.format(self.model_name, self.time, sample, hand))

                for idx in range(3):

                    if not os.path.exists('results/img_results/{}/{}/stft/{:04d}_{}/epoch_{:04d}'.format(self.model_name, self.time, sample, hand, epoch)):
                        os.mkdir('results/img_results/{}/{}/stft/{:04d}_{}/epoch_{:04d}'.format(self.model_name, self.time, sample, hand, epoch))

                    log_spec = tf.math.log(tf.transpose(Zxx[sample][idx]))
                    height = 40
                    width = log_spec.shape[1]
                    x_axis = tf.linspace(0, 2, num=width)
                    y_axis = range(height)
                    plt.pcolormesh(x_axis, y_axis, log_spec[:40, ])
                    plt.title('STFT Magnitude for channel {} for {} hand in iteration {}'.format(channels[idx], hand, epoch))
                    plt.ylabel('Frequency [Hz]')
                    plt.xlabel('Time [sec]')
                    plt.savefig('results/img_results/{}/{}/stft/{:04d}_{}/epoch_{:04d}/{}_stft.png'.format(self.model_name, self.time, sample, hand, epoch, channels[idx]))
                    plt.close()

                observe_signal = sigs[sample, 0, :, :]
                observe_signal = tf.reshape(observe_signal, shape = [500])

                plt.figure(figsize=(20,8))
                plt.plot(observe_signal)
                plt.title('First generated brain signal point for {} hand from sample {} in iteration {}'.format(hand, sample, epoch))
                plt.ylabel('SLORETA(micro voltage)')
                plt.xlabel('Time [sec]')

                if not os.path.exists('results/img_results/{}/{}/signal/{:04d}_{}/epoch_{:04d}'.format(self.model_name, self.time, sample, hand, epoch)):
                    os.mkdir('results/img_results/{}/{}/signal/{:04d}_{}/epoch_{:04d}'.format(self.model_name, self.time, sample, hand, epoch))

                plt.savefig('results/img_results/{}/{}/signal/{:04d}_{}/epoch_{:04d}/signal.png'.format(self.model_name, self.time, sample, hand, epoch))
                plt.close()