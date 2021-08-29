
import os
import mne
import numpy as np

LEFT_VERTEX_SIZE = 1022
RIGHT_VERTEX_SIZE = 1067
SUBGROUP_SIZE = 4

def boolean_brain():

    path = ''

    
    pass

def restore_brain_activation(data):

    left_brain_activation = np.asarray([])
    right_brain_activation = np.asarray([])

    (left_boolean, right_boolean) = boolean_brain()

    for idx in range(LEFT_VERTEX_SIZE):
        pass

    for idx in range(RIGHT_VERTEX_SIZE):
        pass

    return (left_brain_activation, right_brain_activation)

def generate_eeg(data, epoch, time, model_name, n_round):
    
    directory1 = 'results/img_results/{}/{}/{}'.format(model_name, time, n_round)
    directory2 = 'results/img_results/{}/{}/{}/eeg_activation'.format(model_name, time, n_round)

    if not os.path.exists(directory1):
        os.mkdir(directory1)
    
    os.mkdir(directory2)

    (left_brain_activation, right_brain_activation) = restore_brain_activation(data)
    

    pass