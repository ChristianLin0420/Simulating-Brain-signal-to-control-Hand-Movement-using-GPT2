

# import tensorflow as tf

# def get_pretrained_classfier(path = '/home/jupyter-ivanljh123/Simulating-Brain-signal-to-control-Hand-Movement-using-GPT2/pretrained/09_0.92'):
#     #load pretrained model
#     model = tf.keras.models.load_model(path)
#     model.trainable = False

#     return model



import mne
import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
import nibabel as nib
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.datasets import fetch_fsaverage
import scipy.io
from scipy.io import loadmat
from scipy.spatial import Delaunay
import PIL
from PIL import Image
import datetime
import tensorflow as tf
import tfplot
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, Flatten, Concatenate, BatchNormalization, Dropout, Input
from keras.layers.merge import concatenate
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import gc

DIRECTORY_PATH = os.getcwd()

def stft_min_max(X):
    Zxx = tf.signal.stft(X, frame_length=256, frame_step=16)
    Zxx = tf.abs(Zxx)

    # print("shape of X: " + str(X.shape))
    # print("shape of Zxx: " + str(Zxx.shape))
    
    X = Zxx[:, :, :, :40]
    X = tf.reshape(X, [X.shape[0], -1, 40])
    X = tf.transpose(X, perm=[0, 2, 1])
    X = tf.expand_dims(X, axis=3)
    
    # min max scaling (per instance)
    original_shape = X.shape
    X = tf.reshape(X, [original_shape[0], -1])
    X_max = tf.math.reduce_max(X, axis=1, keepdims=True)
    X_min = tf.math.reduce_min(X, axis=1, keepdims=True)
    X = tf.math.divide(tf.math.subtract(X, X_min), tf.math.subtract(X_max, X_min))
    X = tf.reshape(X, original_shape)

    # print("shape of X: " + str(X.shape))

    return X

@tfplot.autowrap()
def plot_spectrogram(data):
    print("data type: {}".format(type(data)))
    fig = tfplot.Figure(figsize=(16, 40), dpi=1)
    plot = fig.add_subplot(111)
    
    log_spec = tf.math.log(data.T)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    print("height : {}".format(height))
    print("width : {}".format(width))
    x_axis = tf.linspace(0, 2, num=width)
    y_axis = tf.range(height)
    plot.pcolormesh(x_axis, y_axis, log_spec)
    plot.axis('off')
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    plt.close(fig)
    # print("fig shape: {}".format(fig))
    return fig

def create_model():
        model = tf.keras.models.Sequential([
            Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), padding='same', activation="selu"),
            BatchNormalization(),
            MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding="valid"),
            Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', activation="selu"),
            BatchNormalization(),
            MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding="valid"),
            Flatten(),
            Dense(50, activation="selu"),
            Dense(1, activation="sigmoid")
        ])
        
        return model

def get_pretrained_classfier_from_path(path = '/home/jupyter-ivanljh123/EEG-forward-and-inverse/models/A09_0.9183/'):
    #load pretrained model
    model = create_model()
    model.load_weights(path)
    model.trainable = False

    return model

def get_pretrained_classfier(shape = None):

    channels_mapping = {
        "EEG-Fz": "Fz",
        "EEG-0": "FC3",
        "EEG-1": "FC1",
        "EEG-2": "FCz",
        "EEG-3": "FC2",
        "EEG-4": "FC4",
        "EEG-5": "C5",
        "EEG-C3": "C3", 
        "EEG-6": "C1",
        "EEG-Cz": "Cz",
        "EEG-7": "C2",
        "EEG-C4": "C4",
        "EEG-8": "C6",
        "EEG-9": "CP3",
        "EEG-10": "CP1",
        "EEG-11": "CPz",
        "EEG-12": "CP2",
        "EEG-13": "CP4",
        "EEG-14": "P1",
        "EEG-Pz": "Pz",
        "EEG-15": "P2",
        "EEG-16": "POz",
        "EOG-left": "EOG-left",
        "EOG-central": "EOG-central",
        "EOG-right": "EOG-right"
    }

    channels_type_mapping = {
        "Fz": "eeg",
        "FC3": "eeg",
        "FC1": "eeg",
        "FCz": "eeg",
        "FC2": "eeg",
        "FC4": "eeg",
        "C5": "eeg",
        "C3": "eeg", 
        "C1": "eeg",
        "Cz": "eeg",
        "C2": "eeg",
        "C4": "eeg",
        "C6": "eeg",
        "CP3": "eeg",
        "CP1": "eeg",
        "CPz": "eeg",
        "CP2": "eeg",
        "CP4": "eeg",
        "P1": "eeg",
        "Pz": "eeg",
        "P2": "eeg",
        "POz": "eeg",
        "EOG-left": "eog",
        "EOG-central": "eog",
        "EOG-right": "eog"
    }

    img = nib.load("brodmann.nii.gz")

    brodmann_data = img.get_fdata()
    brodmann_motor = brodmann_data.reshape(-1) == 4
    # print(brodmann_motor)

    shape, affine = img.shape[:3], img.affine
    coords = np.array(np.meshgrid(*(range(i) for i in shape), indexing='ij'))
    coords = np.rollaxis(coords, 0, len(shape) + 1)
    mm_coords = nib.affines.apply_affine(affine, coords)

    def in_hull(p, hull):
        """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """
        if not isinstance(hull,Delaunay):
            hull = Delaunay(hull)

        return hull.find_simplex(p)>=0

    # my_left_points = None
    # my_right_points = None

    """"
    labels utility function
    """
    def load_subject_labels(name="A01E.mat", dir="drive/Shareddrives/Motor Imagery/BCI competition IV dataset/2a/2a true_labels/"):
        data = scipy.io.loadmat(dir + name)["classlabel"].reshape(-1)
        return data

    def load_all_true_labels(dataset_path):
        data = {}
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                data[file] = load_subject_labels(name=file, dir=root) 
        return data

    """
    load data function
    """
    def load_subject(name="A01T.gdf", dir='drive/Shareddrives/Motor Imagery/BCI competition IV dataset/2a/BCICIV_2a_gdf/', debug=None):
        subject_data = {}
        # Load data
        raw = mne.io.read_raw_gdf(dir + name)
        # Rename channels
        raw.rename_channels(channels_mapping)
        # Set channels types
        raw.set_channel_types(channels_type_mapping)
        # Set montage
        # Read and set the EEG electrode locations
        ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(ten_twenty_montage)
        # Set common average reference
        raw.set_eeg_reference('average', projection=True, verbose=False)
        # Drop eog channels
        raw.drop_channels(["EOG-left", "EOG-central", "EOG-right"])

        subject_data["raw"] = raw
        subject_data["info"] = raw.info
        
        """
        '276': 'Idling EEG (eyes open)'
        '277': 'Idling EEG (eyes closed)'
        '768': 'Start of a trial'
        '769': 'Cue onset left (class 1)'
        '770': 'Cue onset right (class 2)'
        '771': 'Cue onset foot (class 3)'
        '772': 'Cue onset tongue (class 4)'
        '783': 'Cue unknown'
        '1023': 'Rejected trial'
        '1072': 'Eye movements'
        '32766': 'Start of a new run'
        """
        custom_mapping = {'276': 276, '277': 277, '768': 768, '769': 769, '770': 770, '771': 771, '772': 772, '783': 783, '1023': 1023, '1072': 1072, '32766': 32766}
        events_from_annot, event_dict = mne.events_from_annotations(raw, event_id=custom_mapping)

        class_info = "Idling EEG (eyes open): " + str(len(events_from_annot[events_from_annot[:, 2]==276][:, 0])) + "\n" + \
                    "Idling EEG (eyes closed): " + str(len(events_from_annot[events_from_annot[:, 2]==277][:, 0])) + "\n" + \
                    "Start of a trial: " + str(len(events_from_annot[events_from_annot[:, 2]==768][:, 0])) + "\n" + \
                    "Cue onset left (class 1): " + str(len(events_from_annot[events_from_annot[:, 2]==769][:, 0])) + "\n" + \
                    "Cue onset right (class 2): " + str(len(events_from_annot[events_from_annot[:, 2]==770][:, 0])) + "\n" + \
                    "Cue onset foot (class 3): " + str(len(events_from_annot[events_from_annot[:, 2]==771][:, 0])) + "\n" + \
                    "Cue onset tongue (class 4): " + str(len(events_from_annot[events_from_annot[:, 2]==772][:, 0])) + "\n" + \
                    "Cue unknown: " + str(len(events_from_annot[events_from_annot[:, 2]==783][:, 0])) + "\n" + \
                    "Rejected trial: " + str(len(events_from_annot[events_from_annot[:, 2]==1023][:, 0])) + "\n" + \
                    "Eye movements: " + str(len(events_from_annot[events_from_annot[:, 2]==1072][:, 0])) + "\n" + \
                    "Start of a new run: " + str(len(events_from_annot[events_from_annot[:, 2]==32766][:, 0]))
        subject_data["class_info"] = class_info


        epoch_data = {"left": [], "right": [], "foot": [], "tongue": [], "unknown": []}
        rejected_trial = events_from_annot[events_from_annot[:, 2]==1023][:, 0]
        class_dict = {"left": 769, "right": 770, "foot": 771, "tongue": 772, "unknown": 783}
        raw_data = raw.get_data() #(22, 672528)
        start = 0                 # cue
        stop = 500                # cue+3.0s

        for event_class, event_id in class_dict.items():
            current_event = events_from_annot[events_from_annot[:, 2]==event_id][:, 0]
            if event_class == "unknown":
                subject_true_labels = true_labels[name[:4]+".mat"]
                class_dict_labels = {1: "left", 2: "right", 3: "foot", 4: "tongue"}
                for i in range(len(current_event)):
                    # exclude artifact
                    if (current_event[i] - 500 != rejected_trial).all():
                        current_event_data = np.expand_dims(np.array(raw_data[:22, current_event[i]+start:current_event[i]+stop]), axis=0)
                        if (epoch_data.get(class_dict_labels[subject_true_labels[i]]) == None).all():
                            epoch_data[class_dict_labels[subject_true_labels[i]]] = current_event_data
                        else:
                            epoch_data[class_dict_labels[subject_true_labels[i]]] = np.append(epoch_data[class_dict_labels[subject_true_labels[i]]], current_event_data, axis=0)
            else:
                for i in range(len(current_event)):
                    # exclude artifact
                    if((current_event[i] - 500 != rejected_trial).all()):
                        epoch_data[event_class].append(np.array(raw_data[:22, current_event[i]+start:current_event[i]+stop]))
                epoch_data[event_class] = np.array(epoch_data[event_class])

        for event_class, event_data in epoch_data.items():
            epoch_data[event_class] = np.array(event_data)

        subject_data["epoch_data"] = epoch_data
            

        return subject_data

    def load_all_subject(dataset_path):
        data = {}
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                data[file] = load_subject(name=file, dir=root) 
        return data

    # Download fsaverage files
    fs_dir = fetch_fsaverage(verbose=True)
    subjects_dir = op.dirname(fs_dir)

    # The files live in:
    subject = 'fsaverage'
    trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
    src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
    bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

    source = mne.read_source_spaces(src)
    left = source[0]
    right = source[1]
    left_pos = left["rr"][left["inuse"]==1]
    right_pos = right["rr"][right["inuse"]==1]
                            
    transformation = mne.read_trans(op.join(fs_dir, "bem", "fsaverage-trans.fif"))

    save_path = op.join(os.getcwd(), "Shared drives", "Motor Imagery", "Source Estimate")

    true_labels_path = "/home/jupyter-ivanljh123/test2/Simulating-Brain-signal-to-control-Hand-Movement-using-GPT2/2a true_labels/"
    true_labels = load_all_true_labels(true_labels_path)
    print(len(true_labels))

    dataset_path = '/home/jupyter-ivanljh123/test2/Simulating-Brain-signal-to-control-Hand-Movement-using-GPT2/BCICIV_2a_gdf/'
    data = load_all_subject(dataset_path)
    print(len(data))

    """
    create mne epochs data structure from numpy array
    merge training and evaluation data
    """
    def create_epochs(data):
        subjects_data = {}

        for subject in data.keys():
            if "E" in subject:
                continue
            epochs_data = {}
            for event in data[subject]["epoch_data"].keys():
                current_event_data = None
                
                if data[subject]["epoch_data"][event].any():
                    current_event_data = data[subject]["epoch_data"][event]
                if data[subject[:3]+"E.gdf"]["epoch_data"][event].any():
                    current_event_data = np.append(current_event_data, data[subject[:3]+"E.gdf"]["epoch_data"][event], axis=0)
                if current_event_data is not None:
                    epochs_data[event] = mne.EpochsArray(current_event_data, data[subject]["info"], verbose=False)

            subjects_data[subject[:3]] = epochs_data

        return subjects_data

    epochs = create_epochs(data)

    """
    create source_activity (only motor region) first by applying an inverse operator to the epochs 
    create reconstructed eeg by applying a forward operator to the source activity acquired earlier
    save both these files to disk
    """
    def apply_inverse_and_forward(epochs):
        global my_left_points, my_right_points
        
        for subject in epochs.keys():
            for event in epochs[subject].keys():
                
                noise_cov = mne.compute_covariance(epochs[subject][event], tmax=0., method=['shrunk', 'empirical'], rank=None, verbose=False)
                fwd = mne.make_forward_solution(epochs[subject][event].info, trans=trans, src=src,
                                bem=bem, eeg=True, meg=False, mindist=5.0, n_jobs=1)
                fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                            use_cps=True)
                leadfield = fwd_fixed['sol']['data']
                inverse_operator = make_inverse_operator(epochs[subject][event].info, fwd, noise_cov, loose=0.2, depth=0.8)
                
                method = "sLORETA"
                snr = 3.
                lambda2 = 1. / snr ** 2
                stc = apply_inverse_epochs(epochs[subject][event], inverse_operator, lambda2,
                                            method=method, pick_ori="normal", verbose=True)
                
                # get motor region points (once)
                if my_left_points is None and my_right_points is None:
                    my_source = stc[0]
                    mni_lh = mne.vertex_to_mni(my_source.vertices[0], 0, mne_subject)
                    #print(mni_lh.shape)
                    mni_rh = mne.vertex_to_mni(my_source.vertices[1], 1, mne_subject)
                    #print(mni_rh.shape)

                    """
                    fig = plt.figure(figsize=(8, 8))
                    ax = fig.add_subplot(projection='3d')
                    ax.scatter(mm_coords.reshape(-1, 3)[brodmann_motor][:, 0], mm_coords.reshape(-1, 3)[brodmann_motor][:, 1], mm_coords.reshape(-1, 3)[brodmann_motor][:, 2], s=15, marker='|')
                    ax.scatter(mni_lh[:, 0], mni_lh[:, 1], mni_lh[:, 2], s=15, marker='_')
                    ax.scatter(mni_rh[:, 0], mni_rh[:, 1], mni_rh[:, 2], s=15, marker='_')
                    ax.set_xlabel('X Label')
                    ax.set_ylabel('Y Label')
                    ax.set_zlabel('Z Label')
                    plt.show()
                    """

                    my_left_points = in_hull(mni_lh, mm_coords.reshape(-1, 3)[brodmann_motor])
                    my_right_points = in_hull(mni_rh, mm_coords.reshape(-1, 3)[brodmann_motor])

                    mni_left_motor = mne.vertex_to_mni(my_source.vertices[0][my_left_points], 0, mne_subject)
                    #print(mni_left_motor.shape)
                    mni_right_motor = mne.vertex_to_mni(my_source.vertices[1][my_right_points], 1, mne_subject)
                    #print(mni_right_motor.shape)

                    """
                    fig = plt.figure(figsize=(8, 8))
                    ax = fig.add_subplot(projection='3d')
                    ax.scatter(mni_lh[:, 0], mni_lh[:, 1], mni_lh[:, 2], s=15, marker='|')
                    ax.scatter(mni_rh[:, 0], mni_rh[:, 1], mni_rh[:, 2], s=15, marker='_')
                    ax.scatter(mni_left_motor[:, 0], mni_left_motor[:, 1], mni_left_motor[:, 2], s=15, marker='o')
                    ax.scatter(mni_right_motor[:, 0], mni_right_motor[:, 1], mni_right_motor[:, 2], s=15, marker='^')
                    ax.set_xlabel('X Label')
                    ax.set_ylabel('Y Label')
                    ax.set_zlabel('Z Label')
                    plt.show()
                    """
                # slice source activity data
                left_hemi_data = []
                right_hemi_data = []
                for source in stc:
                    left_hemi_data.append(source.data[:len(source.vertices[0])][my_left_points])
                    right_hemi_data.append(source.data[-len(source.vertices[1]):][my_right_points])
                left_hemi_data = np.array(left_hemi_data)
                right_hemi_data = np.array(right_hemi_data)
                
                source_activity_path = op.join(DIRECTORY_PATH, "data", "source activity")
                if not op.exists(source_activity_path):
                    os.makedirs(source_activity_path)
                np.savez_compressed(op.join(source_activity_path, subject+"_"+event+".npz"), data=np.append(left_hemi_data, right_hemi_data, axis=1))
                #source_activity[subject][event] = np.append(left_hemi_data, right_hemi_data, axis=1)
                
                #print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)
                #print(stc[0].data.shape)
                
                # slice reconstructed eeg data
                reconstructed_eeg_data = []
                for source in stc:
                    motor_source = np.zeros_like(source.data)
                    motor_source[:len(source.vertices[0])][my_left_points] = source.data[:len(source.vertices[0])][my_left_points]
                    motor_source[-len(source.vertices[1]):][my_right_points] = source.data[-len(source.vertices[1]):][my_right_points]
                    motor_eeg = np.dot(leadfield, motor_source)
                    reconstructed_eeg_data.append(motor_eeg)
                
                reconstructed_eeg_path = op.join(DIRECTORY_PATH, "data", "reconstructed eeg")
                if not op.exists(reconstructed_eeg_path):
                    os.makedirs(reconstructed_eeg_path)
                np.savez_compressed(op.join(reconstructed_eeg_path, subject+"_"+event+".npz"), data=np.array(reconstructed_eeg_data))
                #reconstructed_eeg[subject][event] = np.array(reconstructed_eeg_data)
                
                del stc, left_hemi_data, right_hemi_data, reconstructed_eeg_data
                gc.collect()

    """
    labels
    left (class 0) right (class 1) foot (class 2) tongue (class 3)

    channels
    c3(7) cz(9) c4(11)
    """

    results = {"A01": {}, "A02": {}, "A03": {}, "A04": {}, "A05": {}, "A06": {}, "A07": {}, "A08": {}, "A09": {}}
    labels = {"left": 0, "right": 1}
    select_channels = [7, 9, 11]
    debug = True
    individual = True
    training = False

    # train model on best two subjects
    data_list = []
    # data_list.append(["A03T.gdf", "A03E.gdf", "A09T.gdf", "A09E.gdf"])
    data_list.append(["A09T.gdf", "A09E.gdf"])

    for data_name in data_list:
        X = None
        Y = None
        for name in data_name:
            for event_class, event_data in data[name]["epoch_data"].items():
                if event_data.size != 0 and event_class in labels:
                    data_samples = None
                    for select_channel in select_channels:
                        data_sample = np.expand_dims(event_data[:, select_channel, :], axis=1)
                        if data_samples is not None:
                            data_samples = np.append(data_samples, data_sample, axis=1)
                        else:
                            data_samples = data_sample
                    
                    event_data = np.array(data_samples)
                    
                    if X is None:
                        X = event_data
                        Y = np.ones(len(event_data), dtype=int) * int(labels[event_class])
                    else:
                        X = np.append(X, event_data, axis=0)
                        Y = np.append(Y, np.ones(len(event_data), dtype=int) * int(labels[event_class]))

        Zxx = tf.signal.stft(X, frame_length=256, frame_step=16)
        Zxx = tf.abs(Zxx)
        # Zxx = Zxx.numpy() 

        # preprocess data
        rgb_weights = tf.constant([0.2989, 0.5870, 0.1140], shape=[3, 1])
        X = None

        # spectrogram
        left_mean_img = {"c3": [], "cZ": [], "c4": []}
        right_mean_img = {"c3": [], "cZ": [], "c4": []}

        # convert stft image to numpy array
        for i in range(Zxx.shape[0]):
            current_image = None
            current_data = Zxx[i][:, :, :40]

            for channel in range(current_data.shape[0]):
                img = plot_spectrogram(current_data[channel])
                img = img[:,:,:3]
                img = tf.cast(img, dtype=tf.float32) / 255

                # convert rgb to gray scale
                img = tf.matmul(img[...,:3], rgb_weights)

                if current_image is None:
                    current_image = img
                else:
                    current_image = np.append(current_image, img, axis=1)

            current_image = np.expand_dims(current_image, axis=0)

            if X is None:
                X = current_image
            else:
                X = np.append(X, current_image, axis=0)

    kfold = 10
    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0

    current_model = None

    skf = StratifiedKFold(n_splits=kfold, shuffle=True)
    skf.get_n_splits(X, Y)
    
    for train_index, test_index in skf.split(X, Y):
        #print(len(train_index), len(test_index))
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        model = create_model()

        log_dir = DIRECTORY_PATH + "/logs/" + data_name[0][:3] + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=200, callbacks=[tensorboard_callback], verbose=0)

        Y_hat = model.predict(X_test)
        Y_hat = (Y_hat >= 0.5)

        new_acc = accuracy_score(Y_test, Y_hat)

        if current_model is None:
            current_model = model
        
        if new_acc > accuracy:
            print("-" * 100)
            print("accuracy: {}, new_accuracy: {}".format(accuracy, new_acc))
            accuracy = new_acc
            current_model = model

    print("accuracy: " + str(new_acc))

    print(type(current_model))
    current_model.trainable = False

    return model