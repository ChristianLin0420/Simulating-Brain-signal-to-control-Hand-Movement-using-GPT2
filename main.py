
import os
import tensorflow as tf

from datetime import datetime

from runner import Runner
from config.config import TrainingConfig
from utils.directory import DirectoryGenerator

if __name__ == '__main__':

    ## get training time
    now = datetime.now()
    time = now.strftime("%d_%m_%Y_%H_%M_%S")

    ## initialize the configuration
    config = TrainingConfig()
    config.save_config(config.model_name, time)
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

    _ = DirectoryGenerator(time, config)

    ## initialize the training runner and start training    
    runner = Runner(config)

    print("Runner starts training!!!")
    runner.run()
    print("Runner finished training!!!")
