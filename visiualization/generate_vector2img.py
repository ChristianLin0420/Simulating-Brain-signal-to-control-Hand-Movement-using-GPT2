
from glob import glob
import os
import glob
import argparse
import imageio

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from model.model_utils import load_model
from config.config_gpt2 import load_model_config
from utils.model_monitor import load_random_vector


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default = 'gpt2gan')
    parser.add_argument("--from_name", default = 1)
    parser.add_argument("--to_name", default = 9)
    parser.add_argument("--steps", default = 20)
    parser.add_argument("--noise_len", default = 784)
    parser.add_argument("--noise_hidden_dim", default = 32)
    parser.add_argument("--model_path", default = None)
    args = parser.parse_args()

    model = str(args.model)
    path = str(args.model_path)
    steps = float(args.steps)

    from_name = str(args.from_name)
    to_name = str(args.to_name)

    noise_len = int(args.noise_len)
    noise_dim = int(args.noise_dim)

    if path is None or not os.path.exists(path):
        print("[Error] Given model path is not valid!")
    else:

        config_filepath = path + ".txt"
        config = load_model_config(config_filepath)

        if config is None:
            print("[Error] Given config file path is not valid!")
        else:

            m_p = path + ".h5"
            from_p = path + '/' + from_name + '.txt'
            to_p = path + '/' + to_name + '.txt'
            
            model = load_model(m_p, model, noise_len, noise_dim)

            from_vector = load_random_vector(from_p, noise_len, noise_dim)
            to_vector = load_random_vector(to_p, noise_len, noise_dim)

            print("from vector shape: {}".format(np.shape(from_vector)))
            print("to vector shape: {}".format(np.shape(to_vector)))

            assert np.shape(from_vector) == np.shape(to_vector)

            delta = (to_vector - from_vector) / steps

            print("doistance between from and to: {}".format(to_vector - from_vector))
            print("doistance between from and to: {}".format(delta))

            # start generating images and visualization of the change 
            now = datetime.now()
            time = now.strftime("%d_%m_%Y_%H_%M_%S")

            directory = path + "/generated_img/" + str(time)

            if not os.path.exists(directory):
                os.mkdir(directory)

            current_step = 0

            while current_step <= int(steps):

                img_path = directory + "/" + str(current_step) + ".png"

                prediction = from_vector + delta * float(current_step)

                _ = plt.figure()

                plt.imshow(prediction[:, :, :, 0] * 127.5 + 127.5, cmap = 'gray')
                plt.axis('off')

                plt.savefig(img_path)
                plt.close()

                current_step += 1

            anim_file = directory + "/generating.gif"

            with imageio.get_writer(anim_file, mode = 'I') as writer:
                filenames = glob.glob(directory + "/*.png")
                filenames = sorted(filenames)

                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)
        

    