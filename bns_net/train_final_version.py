import keras
import numpy as np
import h5py
import os
import json
import load_data
import generator as g
from data_object import DataSet
from aux_functions import get_store_path
from evaluate_nets import evaluate_training
import time
from keras import backend as K
from custom_layers import custom_loss, loss_c1
import tensorflow as tf
import custom_callbacks as cc
from keras.callbacks import ModelCheckpoint

filter_size = (1, 2, 3)
NUM_OF_DETECTORS = 2

def set_filter_size(tup):
    global filter_size
    if isinstance(tup, tuple) and len(tup) == 3:
        filter_size = tup
    return

def incp_lay(x, filter_num):
    global filter_size
    active_filter_sizes = filter_size
    l = keras.layers.Conv1D(3 * filter_num, active_filter_sizes[0], padding='same', activation='relu')(x)
    lm_1 = keras.layers.Conv1D(int(round(filter_num / 2.0)), 1, activation='relu')(x)
    lm_2 = keras.layers.Conv1D(2 * filter_num, active_filter_sizes[1], padding='same', activation='relu')(lm_1)
    rm_1 = keras.layers.Conv1D(int(round(filter_num / 2.0)), 1, activation='relu')(x)
    rm_2 = keras.layers.Conv1D(filter_num, active_filter_sizes[2], padding='same', activation='relu')(rm_1)
    r_1 = keras.layers.MaxPooling1D(4, strides=1, padding='same')(x)
    r_2 = keras.layers.Conv1D(int(round(filter_num)), 1, activation='relu')(r_1)
    
    outp = keras.layers.concatenate([l, lm_2, rm_2, r_2])
    
    return(outp)

def combine_stack(in1, in2):
    l1 = incp_lay(in1, 32)
    lb1 = keras.layers.BatchNormalization()(l1)
    l2 = incp_lay(lb1, 32)
    lb2 = keras.layers.BatchNormalization()(l2)
    l_add = keras.layers.Add()([lb1, lb2])
    
    r1 = incp_lay(in2, 32)
    rb1 = keras.layers.BatchNormalization()(r1)
    r2 = incp_lay(rb1, 32)
    rb2 = keras.layers.BatchNormalization()(r2)
    r_add = keras.layers.Add()([rb1, rb2])
    
    out = keras.layers.concatenate([l_add, r_add])
    
    return(out)

def get_input(input_names, NUM_OF_DETECTORS=2):
    inp_sig = keras.layers.Input(shape=(2048, NUM_OF_DETECTORS), name=input_names[0])
    inp_noi = keras.layers.Input(shape=(2048, NUM_OF_DETECTORS), name=input_names[1])
    add = keras.layers.Add()([inp_sig, inp_noi])
    return((add, inp_sig, inp_noi))

def preprocess(inp):
    DROPOUT_RATE = 0.25
    #Preprocessing
    batch_1 = keras.layers.BatchNormalization()(inp)
    drop_1 = keras.layers.Dropout(DROPOUT_RATE)(batch_1)
    conv_1 = keras.layers.Conv1D(64, 32)(drop_1)
    bn_conv_1 = keras.layers.BatchNormalization()(conv_1)
    act_conv_1 = keras.layers.Activation('relu')(bn_conv_1)
    pool_conv_1 = keras.layers.MaxPooling1D(4)(act_conv_1)
    conv_2 = keras.layers.Conv1D(128, 16)(pool_conv_1)
    bn_conv_2 = keras.layers.BatchNormalization()(conv_2)
    act_conv_2 = keras.layers.Activation('relu')(bn_conv_2)
    return(act_conv_2)

def tcn_residual_block(inp, k=3, d=1, num_filters=32, dropout_rate=0.1, name=None):
    global NUM_OF_DETECTORS
    res = keras.layers.Conv1D(num_filters, 1)(inp)
    
    for _ in range(k-1):
        if _ == 0:
            x = keras.layers.Conv1D(num_filters, 2, dilation_rate=d, padding="causal")(inp)
        else:
            x = keras.layers.Conv1D(num_filters, 2, dilation_rate=d, padding="causal")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(dropout_rate)(x)
    
    if name == None:
        out = keras.layers.Add()([x, res])
    else:
        out = keras.layers.Add(name=name)([x, res])
    
    return(out)

def get_tcn_stack(inp, name="TCN"):
    global NUM_OF_DETECTORS
    kernel = 3
    depth = 11
    
    for i in range(depth):
        if i == 0:
            x = tcn_residual_block(inp, k=kernel, d=2**i)
        else:
            if i == depth-1:
                x = tcn_residual_block(x, k=kernel, d=2**i, name=name, num_filters=2)
            else:
                x = tcn_residual_block(x, k=kernel, d=2**i)
    
    return(x)

def aux_out(inp, name):
    pool = keras.layers.AveragePooling1D(8)(inp)
    dim_red = keras.layers.Conv1D(16, 1)(pool)
    flat = keras.layers.Flatten()(dim_red)
    out = keras.layers.Dense(1, name=name, activation='relu')(flat)
    return out

def get_model():
    NUM_DETECTORS = 2
    NUM_CHANNELS = 8
    DROPOUT_RATE = 0.25
    FILTER_NUM = 32
    
    inp1, is1, in1 = get_input(('Input_signal_1', 'Input_noise_1'), NUM_OF_DETECTORS=NUM_DETECTORS)
    inp2, is2, in2 = get_input(('Input_signal_2', 'Input_noise_2'), NUM_OF_DETECTORS=NUM_DETECTORS)
    inp3, is3, in3 = get_input(('Input_signal_3', 'Input_noise_3'), NUM_OF_DETECTORS=NUM_DETECTORS)
    inp4, is4, in4 = get_input(('Input_signal_4', 'Input_noise_4'), NUM_OF_DETECTORS=NUM_DETECTORS)
    inp5, is5, in5 = get_input(('Input_signal_5', 'Input_noise_5'), NUM_OF_DETECTORS=NUM_DETECTORS)
    inp6, is6, in6 = get_input(('Input_signal_6', 'Input_noise_6'), NUM_OF_DETECTORS=NUM_DETECTORS)
    inp7, is7, in7 = get_input(('Input_signal_7', 'Input_noise_7'), NUM_OF_DETECTORS=NUM_DETECTORS)
    inp8, is8, in8 = get_input(('Input_signal_8', 'Input_noise_8'), NUM_OF_DETECTORS=NUM_DETECTORS)
    
    #TCN-part
    tcn1 = get_tcn_stack(inp1, name="TCN_out_1s_part_1")
    tcn2 = get_tcn_stack(inp2, name="TCN_out_1s_part_2")
    tcn3 = get_tcn_stack(inp3, name="TCN_out_2s")
    tcn4 = get_tcn_stack(inp4, name="TCN_out_4s")
    tcn5 = get_tcn_stack(inp5, name="TCN_out_8s")
    tcn6 = get_tcn_stack(inp6, name="TCN_out_16s")
    tcn7 = get_tcn_stack(inp7, name="TCN_out_32s")
    tcn8 = get_tcn_stack(inp8, name="TCN_out_64s")
    
    #Add TCN to input
    add1 = keras.layers.Add()([inp1, tcn1])
    add2 = keras.layers.Add()([inp2, tcn2])
    add3 = keras.layers.Add()([inp3, tcn3])
    add4 = keras.layers.Add()([inp4, tcn4])
    add5 = keras.layers.Add()([inp5, tcn5])
    add6 = keras.layers.Add()([inp6, tcn6])
    add7 = keras.layers.Add()([inp7, tcn7])
    add8 = keras.layers.Add()([inp8, tcn8])
    
    #Maybe this preprocessing can be reduced and one just needs to add
    #BatchNormalization and Dropout, but no convolution layers.
    lvl_1_1 = preprocess(add1)
    lvl_1_2 = preprocess(add2)
    lvl_1_3 = preprocess(add3)
    lvl_1_4 = preprocess(add4)
    lvl_1_5 = preprocess(add5)
    lvl_1_6 = preprocess(add6)
    lvl_1_7 = preprocess(add7)
    lvl_1_8 = preprocess(add8)
    
    lvl_2_1 = combine_stack(lvl_1_1, lvl_1_2)
    lvl_2_2 = combine_stack(lvl_1_3, lvl_1_4)
    lvl_2_3 = combine_stack(lvl_1_5, lvl_1_6)
    lvl_2_4 = combine_stack(lvl_1_7, lvl_1_8)
    
    aux_11 = aux_out(lvl_2_1, 'Aux_11')
    aux_12 = aux_out(lvl_2_2, 'Aux_12')
    aux_13 = aux_out(lvl_2_3, 'Aux_13')
    aux_14 = aux_out(lvl_2_4, 'Aux_14')
    
    lvl_3_1 = combine_stack(lvl_2_1, lvl_2_2)
    lvl_3_2 = combine_stack(lvl_2_3, lvl_2_4)
    
    aux_21 = aux_out(lvl_3_1, 'Aux_21')
    aux_22 = aux_out(lvl_3_2, 'Aux_22')
    
    lvl_4_1 = combine_stack(lvl_3_1, lvl_3_2)
    
    pool_2 = keras.layers.MaxPooling1D(4)(lvl_4_1)
    dim_red = keras.layers.Conv1D(32, 1)(pool_2)
    flatten = keras.layers.Flatten()(dim_red)
    
    dense_1 = keras.layers.Dense(2)(flatten)
    dense_2 = keras.layers.Dense(1, activation='relu', name='Out_SNR')(dense_1)
    
    dense_3 = keras.layers.Dense(3)(flatten)
    dense_4 = keras.layers.Dense(2, activation='softmax', name='Out_Bool')(dense_3)
    
    model = keras.models.Model(inputs=[is1, in1, is2, in2, is3, in3, is4, in4, is5, in5, is6, in6, is7, in7, is8, in8], outputs=[dense_2, dense_4, tcn1, tcn2, tcn3, tcn4, tcn5, tcn6, tcn7, tcn8, aux_11, aux_12, aux_13, aux_14, aux_21, aux_22])
    
    return(model)

class DataGenerator(keras.utils.Sequence):
    def __init__(self, signals, noise, signal_labels, index_list,
                 noise_snr=4., batch_size=32, shuffle=True,
                 label_type='all', use_inputs=None):
        """A Keras generator that generates the input and labels for
        seperate lists of signals and noise instances.
        
        Arguments
        ---------
        signals : list of np.array
            A list of all signals that may be used.
        noise : list of np.array
            A list of all noises that may be used.
        signal_labels : list of float
            A list of the associated SNRs to the signals.
        index_list : list of tuples of 2 ints
            A list containing tuples of length 2. The entries of the
            tuple are the indices to the signal and noise lists
            respectively. Setting the singal index to -1 has the effect
            of yielding a pure noise sample.
        noise_snr : {float, 4.}
            The SNR attributed to a pure noise sample.
        batch_size : {int, 32}
            The size of each training batch. (Number of samples)
        shuffle : {bool, True}
            Shuffle the dataset for each invocation.
        label_type : {str, 'all'}
            Which output the label should be. Must be either
            'snr_and_psc' (gets the SNR and p-score as output) or
            'signal' (gets the pure signal as output) or 'snr' (get just
            the SNR as output) or 'all' (get the specific output form
            needed for the network of the paper).
        use_inputs : {list of int of None, None}
            Which inputs should be used. If None is given all inputs
            will be used. Otherwise the list will correspond to the
            index of the sample rate. (e.g. if only the two parts of
            duration 0.5s at a sample rate of 4096Hz should be used
            provide "use_inputs=[0, 1]") The highest number in the list
            has to be smaller than 8.
        """
        self.signals = signals
        self.noise = noise
        self.signal_labels = signal_labels
        self.index_list = index_list
        self.noise_snr = noise_snr
        self.batch_size = batch_size
        self.shuffle = shuffle
        #ATTENTION: Changed this to fit the three channels
        if label_type in ['snr_and_psc', 'signals', 'snr', 'all']:
            self.label_type = label_type
        else:
            raise ValueError('Unknown label_type {}.'.format(label_type))
        self.use_inputs = list(range(8)) if use_inputs is None else use_inputs
        self.on_epoch_end()
    
    def __len__(self):
        return(int(np.ceil(float(len(self.index_list)) / self.batch_size)))
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.index_list))
        if self.shuffle:
            np.random.shuffle(self.indices)
            self.indices = list(self.indices)
        else:
            self.indices = list(self.indices)
    
    def __getitem__(self, index):
        if (index + 1) * self.batch_size >= len(self.indices):
            indices = self.indices[index*self.batch_size:]
        else:
            indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        X, y = self.__data_generation(indices)
        
        return(X, y)
    
    def __data_generation(self, indices):
        num_detectors = 2
        num_channels = len(self.use_inputs)
        num_inputs = 2 * num_channels #2 comes from the split of signal and noise
        X = [np.zeros([len(indices), num_detectors, 2048]) for i in range(num_inputs)]
        
        y_1 = np.zeros((len(indices), 1))
        
        y_2 = np.zeros((len(indices), 2))
        
        for i, idx in enumerate(indices):
            sig_ind, noi_ind = self.index_list[idx]
                        
            for j, inp in enumerate(self.use_inputs):
                if not sig_ind == -1:
                    if inp == 0:
                        X[2*j][i][0] = self.signals[sig_ind].transpose()[0][2048:]
                        X[2*j][i][1] = self.signals[sig_ind].transpose()[7][2048:]
                    else:
                        X[2*j][i][0] = self.signals[sig_ind].transpose()[inp-1][:2048]
                        X[2*j][i][1] = self.signals[sig_ind].transpose()[inp+6][:2048]
                
                if inp == 0:
                    X[2*j+1][i][0] = self.noise[noi_ind].transpose()[0][:2048]
                    X[2*j+1][i][1] = self.noise[noi_ind].transpose()[7][:2048]
                else:
                    X[2*j+1][i][0] = self.noise[noi_ind].transpose()[inp-1][:2048]
                    X[2*j+1][i][1] = self.noise[noi_ind].transpose()[inp+6][:2048]
            
            if not sig_ind == -1:
                y_1[i] = self.signal_labels[sig_ind]
                
                y_2[i][0] = 1.0
                y_2[i][1] = 0.0
            else:
                y_1[i] = self.noise_snr
                
                y_2[i][0] = 0.0
                y_2[i][1] = 1.0
            
        
        X = [dat.transpose(0, 2, 1) for dat in X]
        
        if self.label_type == 'snr_and_psc':
            return((X, [y_1, y_2]))
        elif self.label_type == 'snr':
            return((X, y_1))
        elif self.label_type == 'signals':
            return((X, [X[2*i] for i in range(len(X) / 2)]))
        elif self.label_type == 'all':
            return((X, [y_1, y_2] + [X[2*i] for i in range(len(X) / 2)] + [y_1 for _ in range(6)]))

def generate_unique_index_pairs_noise(noise, num_pairs):
    ret = np.random.choice(len(noise), size=num_pairs, replace=False)
    np.random.shuffle(ret)
    ret = [(-1, pt) for pt in ret]
    return(ret)

def generate_unique_index_pairs(signals, noise, num_pairs, generate_signals_only=True, noise_index_range=None, signal_index_range=None):
    if noise_index_range == None:
        noise_index_range = [0, len(noise)]
    elif not isinstance(noise_index_range, list) and not isinstance(noise_index_range, tuple) and len(noise_index_range) == 2:
        raise ValueError('noise_index_range needs to be a list or tuple of length 2.')
    
    if signal_index_range == None:
        signal_index_range = [0, len(signals)]
    elif not isinstance(signal_index_range, list) and not isinstance(signal_index_range, tuple) and len(signal_index_range) == 2:
        raise ValueError('signal_index_range needs to be a list or tuple of length 2.')
    
    len_noi = noise_index_range[1] - noise_index_range[0]
    len_sig = signal_index_range[1] - signal_index_range[0]
    if len_sig < 0 or len_noi < 0 or noise_index_range[0] < 0 or signal_index_range[0] < 0 or noise_index_range[1] > len(noise) or signal_index_range[1] > len(signals):
        raise ValueError('start indices for signals and noise must be within the range of signals and noise.')
    if generate_signals_only:
        max_len = len_sig * len_noi
    else:
        max_len = (len_sig+1) * len_noi
        
    if max_len < num_pairs:
        raise ValueError('Tried to generate {} unique pairs from {} possible combinations.'.format(num_pairs, max_len))
    
    #Check if it is more efficient to pick pairs to not include
    inv = False
    if num_pairs > (len_sig * len_noi) / 2:
        inv = True
    
    curr_pairs = 0
    max_pair = max_len - num_pairs if inv else num_pairs
    poss = np.zeros(max_len, dtype=int)
    while curr_pairs < max_pair:
        r_int = np.random.randint(0, max_len)
        if poss[r_int] == 0:
            poss[r_int] = 1
            curr_pairs += 1
    
    true_val = 0 if inv else 1
    
    ret = []
    
    for i, val in enumerate(poss):
        if val == true_val:
            sig_idx = i / len_noi
            noi_idx = i % len_noi

            if sig_idx == len_sig:
                ret.append((-1, noi_idx))
            else:
                ret.append((sig_idx, noi_idx))
    
    ret = np.array(ret, dtype=int)
    np.random.shuffle(ret)
    
    ret = [(pt[0]+signal_index_range[0], pt[1]+noise_index_range[0]) for pt in ret]
    
    return(ret)

def main():
    #Training stats
    epochs = 50
    
    #Names and paths
    name = 'final_network_retrain'
    net_path = os.path.join(get_store_path(), name)
    if not os.path.isdir(net_path):
        os.mkdir(net_path)
    templates_path = os.path.join(get_store_path(), 'templates_correct_whiten.hf5')
    
    #Parameters
    num_training_signals = 56250
    num_training_noise = 161250
    num_validation_signals = 18750
    num_validation_noise = 53750
    noise_snr = 4.
    
    #Load the data
    print("Loading data...")
    with h5py.File(templates_path, 'r') as f:
        training_signal_labels = f['signals/snr'][:num_training_signals]
        training_signals = f['signals/data'][:num_training_signals]
        training_noise = f['noise/data'][:num_training_noise]
        
        validation_signal_labels = f['signals/snr'][num_training_samples:num_training_signals+num_validation_signals]
        validation_signals = f['signals/data'][num_training_samples:num_training_signals+num_validation_signals]
        validation_noise = f['noise/data'][num_training_noise:num_training_noise+num_validation_noise]
    
    print("Generating indices for training set...")
    training_noise_indices = generate_unique_index_pairs_noise(training_noise, num_training_noise)
    training_signal_indices = generate_unique_index_pairs(training_signals, training_noise, num_training_noise)
    training_index_list = training_noise_indices + training_signal_indices
    np.random.shuffle(training_index_list)
    
    print("Generating indices for validation set...")
    validation_noise_indices = generate_unique_index_pairs_noise(validation_noise, num_validation_noise)
    validation_signal_indices = generate_unique_index_pairs(validation_signals, validation_noise, num_training_noise)
    validation_index_list = validation_noise_indices + validation_signal_indices
    np.random.shuffle(validation_index_list)
    
    print("Setting up generators...")
    training_generator = DataGenerator(training_signals,
                                       training_noise,
                                       training_signal_labels,
                                       training_index_list,
                                       noise_snr=noise_snr,
                                       batch_size=16,
                                       shuffle=True,
                                       label_type='all')
    validation_generator = DataGenerator(validation_signals,
                                         validation_noise,
                                         validation_signal_labels,
                                         validation_index_list,
                                         noise_snr=noise_snr,
                                         batch_size=16,
                                         shuffle=False,
                                         label_type='all')
    
    #Setting up the model for training
    print("Setting up model...")
    model = get_model()
    
    print("Compiling model...")
    aux_weight = 0.1
    model.compile(loss=['mse', 'categorical_crossentropy', 'mse', 'mse', 'mse', 'mse', 'mse', 'mse', 'mse', 'mse', 'mse', 'mse', 'mse', 'mse', 'mse', 'mse'], loss_weights=[1.0, 0.5, aux_weight, aux_weight, aux_weight, aux_weight, aux_weight, aux_weight, aux_weight, aux_weight, aux_weight, aux_weight, aux_weight, aux_weight, aux_weight, aux_weight], optimizer='adam', metrics={'Out_SNR': 'mape', 'Out_Bool': 'accuracy'})
    
    #Setting up callbacks
    print("Setting up callbacks...")
    SensTracker = cc.SensitivityTracker(validation_generator, net_path, interval=1)
    check_path = os.path.join(net_path, name + '_epoch_{epoch:d}.hf5')
    ModelCheck = ModelCheckpoint(check_path, verbose=1, period=1)
    
    #Training
    print("Starting to train...")
    hist = model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=epochs, max_q_size=2, callbacks=[ModelCheck, SensTracker])
    
    print("Done training, saving history to file...")
    with open(os.path.join(net_path, 'history.json'), 'w') as f:
        json.dump(hist.history, f)
    
    print("Finished, returning")
    return

if __name__ == '__main__':
    main()
