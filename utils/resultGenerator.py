
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from res.brain import Brain

from .brain_activation import (
    boolean_brain, 
    fetch_brain_template,
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
        self.brain_template = fetch_brain_template()

    def generate_figure(self, data, label, title, path, flag = False):

        if flag is False:
            lst_iter = np.arange(self.config.epochs)
        else:
            lst_iter = np.arange(data.shape[1])

        if data.ndim == 3:
            data_count = data.shape[0]
        elif data.ndim == 2:
            data_count = 1
            data = np.expand_dims(data, axis = 0)
        else:
            print("[Error] resultGenerator gets invalid data with dimension not equals to 2 or 3!!!")

        for i in range(data_count):
            mean = np.mean(data[i], axis = 0)
            standrad_ev = np.std(data[i], axis = 0)

            plt.figure(figsize = (12, 6))
            plt.plot(lst_iter, mean, label = label[i], linewidth = 1)
            plt.fill_between(lst_iter, mean - standrad_ev, mean + standrad_ev, alpha = 0.2)

        plt.xlabel("number of iterations")
        plt.legend(loc = 'upper left')
        plt.title(title)

        # save image
        plt.savefig(path)  
        plt.close()

    def generate_compare_results_figure(self, model_names, times, compare_parameter: str = None):

        if not os.path.exists("result/{}/compare_results".format(model_names[0])):
            os.mkdir("result/{}/compare_results".format(model_names[0]))

        if compare_parameter is None:
            print("[Error] compare_parameter was not given!!!")

        if len(model_names) == 1 and len(times) > 1:
            filenames = [f for f in os.listdir("result/{}/{}/history/1/".format(model_names[0], times[0])) if f.endswith(".json")]

            for filename in filenames:
                labels = list()
                datas = list()

                for time in times:
                    data = list()

                    config_path = "result/{}/{}/config/config.json".format(model_names[0], time)
                    cd = json.load(config_path)
                    label = "{}_{}".format(compare_parameter, cd[compare_parameter])

                    if label is None:
                        print("[Error] compare_parameter cannot fit to any key of the config file!!!")
                    else:
                        labels.append(label)

                        for r in self.config.rounds:
                            path = "result/{}/{}/history/{}/{}".format(model_names[0], time, r, filename)
                            d = json.load(open(path))
                            d = data[filename[:-5]]
                            data.append(d)

                datas.append(data)
                datas = np.asarray(datas)
                self.generate_figure(   datas, labels, 
                                        "{}_{}_with_{}".format(model_names[0], filename[:-5], compare_parameter), 
                                        "result/{}/compare_results/{}_{}_with_{}.png".format(model_names[0], model_names[0], filename[:-5], compare_parameter))

        elif len(model_names) > 1 and len(times) > 1:
            pass

    def generate_training_result_figure(self):
        filenames = [f for f in os.listdir("result/{}/{}/history/1/".format(self.config.model_name, self.time)) if f.endswith(".json")]
        print(filenames)

        for filename in filenames:
            datas = []

            for round in range(self.config.rounds):
                path = "result/{}/{}/history/{}/{}".format(self.config.model_name, self.time, round, filename)
                if os.path.exists(path):
                    data = json.load(open(path))
                    data = data[filename[:-5]]
                    datas.append(data)

            datas = np.asarray(datas)
            self.generate_figure(datas, [filename[:-5]], filename[:-5], "result/{}/{}/figure/{}.png".format(self.config.model_name, self.time, filename))

    def generate_all_channels_eeg(self, data, epoch, round):
        channels = Brain().get_channel_names()
        real_data = np.asarray(self.real_data)
        t_matrix = np.asarray(self.transformation_matrix)

        generated_num_per_class = int(self.config.example_to_generate / self.config.class_count)

        for i in range(self.config.class_count):
            class_data = np.mean(data[i * generated_num_per_class : (i + 1) * generated_num_per_class], axis = 0)
            class_data = np.reshape(class_data, class_data.shape[:2]).tolist()
            left, right = restore_brain_activation(class_data, self.boolean_l, self.boolean_r)
            left = np.asarray(left)
            right = np.asarray(right)
            class_data = np.concatenate([left, right], axis = 0)
            class_data = np.dot(t_matrix, class_data)
            real_converted_data = np.dot(t_matrix, real_data[i])

            if not os.path.exists("result/{}/{}/eeg/{}/epoch_{:04d}".format(self.config.model_name, self.time, round, epoch)):
                os.mkdir("result/{}/{}/eeg/{}/epoch_{:04d}".format(self.config.model_name, self.time, round, epoch))

            for c in range(real_converted_data.shape[0]):
                wave1 = np.expand_dims(real_converted_data[c], axis = 0)
                wave2 = np.expand_dims(class_data[c], axis = 0)
                waves = np.concatenate([wave1, wave2], axis = 0)
                labels = ["real_data_channel_{}".format(channels[c]), "generated_data_channel_{}".format(channels[c])]

                self.generate_figure(   waves, 
                                        labels, 
                                        "class_{}_channel_{}_epoch_{}".format(i, channels[c], epoch), "result/{}/{}/eeg/{}/epoch_{:04d}/class_{}_{}.png".format(self.config.model_name, self.time, round, epoch, i, channels[c]), 
                                        True)

        data = None
        del data

    def generate_topographs(self, data, epoch, round):
        real_data = np.asarray(self.real_data)
        t_matrix = np.asarray(self.transformation_matrix)

        generated_num_per_class = int(self.config.example_to_generate / self.config.class_count)

        for i in range(self.config.class_count):
            class_data = np.mean(data[i * generated_num_per_class : (i + 1) * generated_num_per_class], axis = 0)
            class_data = np.reshape(class_data, class_data.shape[:2]).tolist()
            left, right = restore_brain_activation(class_data, self.boolean_l, self.boolean_r)
            left = np.asarray(left)
            right = np.asarray(right)
            class_data = np.concatenate([left, right], axis = 0)
            class_data = np.dot(t_matrix, class_data)
            real_converted_data = np.dot(t_matrix, real_data[i])

            if not os.path.exists("result/{}/{}/topography/{}/epoch_{:04d}".format(self.config.model_name, self.time, round, epoch)):
                os.mkdir("result/{}/{}/topography/{}/epoch_{:04d}".format(self.config.model_name, self.time, round, epoch))

            self.brain_template.data = class_data
            ax = self.brain_template.plot_topomap(times = np.linspace(0.0, 0.2, 20), ch_type = 'eeg', time_unit='s', ncols=5, nrows='auto', title = 'Original Class_{} Brain Activation in iteration {}'.format(i, epoch), show = False)
            ax.savefig("result/{}/{}/topography/{}/epoch_{:04d}/original_class_{}_topograph.png".format(self.config.model_name, self.time, round, epoch, i))
            plt.close(ax)

            self.brain_template.data = real_converted_data
            ax = self.brain_template.plot_topomap(times = np.linspace(0.0, 0.2, 20), ch_type = 'eeg', time_unit='s', ncols=5, nrows='auto', title = 'Generated Class_{} Brain Activation in iteration {}'.format(i, epoch), show = False)
            ax.savefig("result/{}/{}/topography/{}/epoch_{:04d}/generated_class_{}_topograph.png".format(self.config.model_name, self.time, round, epoch, i))
            plt.close(ax)

        data = None
        del data


    def generate_stft(self, data, epoch, _round):
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
        
        bz = int(data.shape[0])
        channels = ["C3", "C4", "Cz"]

        ## generate short-time fourier transform figures
        for sample in range(bz):
            if not os.path.exists('result/{}/{}/stft/{}/epoch_{:04d}'.format(self.config.model_name, self.time, _round, epoch)):
                os.mkdir('result/{}/{}/stft/{}/epoch_{:04d}'.format(self.config.model_name, self.time, _round, epoch))

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
                plt.savefig('result/{}/{}/stft/{}/epoch_{:04d}/class_{}_{}.png'.format(self.config.model_name, self.time, _round, epoch, sample + 1, channels[idx]))
                plt.close()

        data = None
        del data