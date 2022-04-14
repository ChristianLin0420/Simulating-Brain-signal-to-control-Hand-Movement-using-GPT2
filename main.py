
from distutils.log import error
import os
import argparse
import tensorflow as tf

from datetime import datetime

from runner import Runner
from config.config import TrainingConfig
from utils.directory import DirectoryGenerator
from utils.resultGenerator import ResultGenerator
from utils.datasetGenerator import (
    DatasetGenerator, 
    get_training_reconstruct_signals, 
    get_training_raw_signals
)

if __name__ == '__main__':

    ## read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default = 'training')  
    parser.add_argument("--model_name", nargs = '+', default = None)
    parser.add_argument("--time", nargs = '+', default = None)
    parser.add_argument("--figure", default = 'accuracy')
    args = parser.parse_args()

    if args.mode == "generate_result":
        if args.time is None:
            print("[Error] No specific time input was given!!!")
        elif len(args.time) > 1:
            print("[Error] More than one time input was given!!!")
        elif args.model_name is None:
            print("[Error] No specific model name input was given!!!")
        elif len(args.model_name) > 1:
            print("[Error] Multiple model names input was given!!!")
        else:
            config_path = "result/{}/{}/config/config.json".format(args.model_name[0], args.time[0])
            
            if not os.path.exists(config_path):
                print("[Error] Given config path is not existed!!!")
            else:
                config = TrainingConfig()
                config.load_config("result/{}/{}/config/config.json".format(args.model_name[0], args.time[0]))
                filenames, labels = get_training_reconstruct_signals()
                raw_filenames, raw_labels = get_training_raw_signals(subject_count = config.subject_count)
                d_generator = DatasetGenerator(filenames = filenames, raw_filenames = raw_filenames, labels = labels, raw_labels = raw_labels, config = config)
                _, _, avg_real_data = d_generator.get_reconstructed_items(filenames, labels)

                generator = ResultGenerator(config, args.time[0], avg_real_data)
                generator.generate_training_result_figure()
                # generator.generate_all_channels_eeg()
                # generator.generate_topographs()
    elif args.mode == "compare_result":
        if args.time is None:
            print("[Error] No specific time input was given!!!")
        elif args.model_name is None:
            print("[Error] No specific model name input was given!!!")
        else:
            pass
            
    elif args.mode == "training":
        ## get training time
        now = datetime.now()
        time = now.strftime("%d_%m_%Y_%H_%M_%S")

        ## initialize the configuration and directory
        config = TrainingConfig()
        _ = DirectoryGenerator(time, config)
        config.save_config(config.model_name[0], time)

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
        generator = ResultGenerator(config, time, runner.real_average_data)
        generator.generate_training_result_figure()
        # generator.generate_all_channels_eeg()
        # generator.generate_topographs()
        print("Finish generating results!!!")
    else:
        error("[Main] given argument is invalid!!!")

