
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from res.brain import Brain

from brain_activation import (
    boolean_brain, 
    transformation_matrix, 
    restore_brain_activation, 
    restore_brain_activation_tf, 
)

class ResultGenerator(object):

    def __init__(self, config, time, real_data):
        self.config = config
        self.time = time
        self.real_data = real_data

        (self.boolean_l, self.boolean_r) = boolean_brain()
        self.transformation_matrix = transformation_matrix()

    def generate_figure(self, data, label, title):
        lst_iter = np.arange(self.config.epochs)

        if data.ndim == 3:
            data_count = data.shape[0]
        elif data.ndim == 2:
            data_count = 1
            data = np.expand_dims(data, axis = 0)
        else:
            print("[Error] resultGenerator gets invalid data with dimension not equals to 2 or 3!!!")

        for i in data_count:
            mean = np.mean(data[i], axis = 0)
            standrad_ev = np.std(data[i], axis = 0)

            plt.figure(figsize = (8,5))
            plt.plot(lst_iter, mean, label = label[i], linewidth = 1)
            plt.fill_between(lst_iter, mean - standrad_ev, mean + standrad_ev, alpha = 0.2)

        plt.xlabel("number of iterations")
        plt.legend(loc = 'upper left')
        plt.title(title)

        # save image
        plt.savefig("result/{}/{}/figure/{}.png".format(self.config.model_name, self.time, title))  
        plt.close()

    def generate_training_result_figure(self):
        filenames = [f for f in os.listdir("result/{}/{}/history/1/".format(self.config.model_name, self.time, self.round)) if f.endswith(".json")]
        print(filenames)

        for filename in filenames:
            datas = []

            for round in self.config.rounds:
                path = "result/{}/{}/history/1/{}".format(self.config.model_name, self.time, round, filename)
                data = json.load(open(path))
                data = data[filename[:-5]]
                datas.append(data)

            datas = np.asarray(datas)
            self.generate_figure(datas, [filename], filename)

    def generate_all_channels_eeg(self):
        channels = Brain.get_channel_names()
        real_data = np.asarray(self.real_data)
        t_matrix = np.asarray(self.transformation_matrix)
        real_converted_data = np.dot(t_matrix, real_data)

        for r in range(self.config.rounds):
            for e in range(0, self.config.epochs, 10):
                filename = "result/{}/{}/generated/{}/{}/generated.json".format(self.config.model_name, self.time, r, e)
                data = json.load(filename)
                data = data["generated"]
                left, right = restore_brain_activation(data, self.boolean_l, self.boolean_r)
                left = np.asarray(left)
                right = np.asarray(right)
                data = np.concatenate([left, right], axis = 0)
                data = np.dot(t_matrix, data)

                for i in range(data.shape[0]):
                    wave1 = np.expand_dims(real_converted_data[i], axis = 0)
                    wave2 = np.expand_dims(data[i], axis = 0)
                    waves = np.concatenate([wave1, wave2])
                    labels = ["real_data_channel_{}".format(channels[i]), "generated_data_channel_{}".format(channels[i])]

                    self.generate_figure(waves, labels, "Channel_{}_epoch_{}".format(channels[i], e))



    def generate_topographs(self):
        pass

    def generate_stft(self, data, epoch):
        brain = None
        signals = None

        for i in range(data.shape[0]):
            (l_tmp, r_tmp) = restore_brain_activation_tf(data[i], self.boolean_l, self.boolean_r)
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