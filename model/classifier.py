

import tensorflow as tf

def get_pretrained_classfier(path = '/home/jupyter-ivanljh123/Simulating-Brain-signal-to-control-Hand-Movement-using-GPT2/pretrained/09_0.92.h5'):
    #load pretrained model
    model = tf.keras.models.load_model(path)
    model.trainable = False

    return model