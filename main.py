
from distutils.log import error
import os
import argparse
import tensorflow as tf

from datetime import datetime

from runner import Runner
from config.config import TrainingConfig
from utils.directory import DirectoryGenerator
from utils.resultGenerator import ResultGenerator

if __name__ == '__main__':

    ## read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default = 'training')  
    parser.add_argument("--model_name", default = None)
    parser.add_argument("--time", default = None)
    parser.add_argument("--figure", default = 'accuracy')
    args = parser.parse_args()

    if args.mode == "build":
        if args.time is None:
            print("[Error] No specific time input was given!!!")
        elif args.model_name is None:
            print("[Error] No specific model name input was given!!!")
        else:
            config_path = "result/{}/{}/config/config.txt".format(args.model_name, args.time)
            
            if not os.path.exists(config_path):
                print("[Error] Given config path is not existed!!!")
            else:
                config = TrainingConfig()
                config.load_config("result/{}/{}/config/config.txt")
                generator = ResultGenerator(config, args.time)

                generator.generate_training_result_figure()
                generator.generate_all_channels_eeg()
                generator.generate_topographs()
            
    elif args.mode == "training":
        ## get training time
        now = datetime.now()
        time = now.strftime("%d_%m_%Y_%H_%M_%S")

        ## initialize the configuration and directory
        config = TrainingConfig()
        _ = DirectoryGenerator(time, config)
        config.save_config(config.model_name, time)

        print("time: {}".format(time))
        print(config)

        ## environment settings
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

        # if use_gpu is TRUE, then get one and run the program on it
        if config.gpu:
            try:
                gpus = tf.config.list_physical_devices(device_type = 'GPU')
                
                if gpus:
                    tf.config.set_visible_devices(devices = gpus[0], device_type = 'GPU')
                    tf.config.experimental.set_memory_growth(gpus[0], True)
            except:
                print("[No GPR] there is no availible gpu to use!!!")

        ## initialize the training runner and start training    
        runner = Runner(config, time)

        print("Runner starts training!!!")
        runner.run()
        print("Runner finished training!!!")

        print("Start generating results!!!")
        generator = ResultGenerator(config, args.time)
        generator.generate_training_result_figure()
        generator.generate_all_channels_eeg()
        generator.generate_topographs()
        print("Finish generating results!!!")
    else:
        error("[Main] given argument is invalid!!!")

