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

def inception_block(inp):
    l1 = incp_lay(inp, 32)
    lb1 = keras.layers.BatchNormalization()(l1)
    l2 = incp_lay(lb1, 32)
    lb2 = keras.layers.BatchNormalization()(l2)
    l_add = keras.layers.Add()([lb1, lb2])
    return l_add

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

def train_tcn(tcn_model, generator, epochs=1):
    tcn_model.compile(optimizer='adam', loss='mse')
    #Somehow train the network here to recreate waveform
    tcn_model.fit_generator(generator, epochs=epochs)
    tcn_model.trainable = False
    return

def train_inception_part(incp_model, generator, epochs=1, inputs=[], name='Inception block'):
    aux_out_part = aux_out(incp_model(inputs), name)
    train_model = keras.models.Model(inputs=inputs, outputs=aux_out_part)
    train_model.compile(optimizer='adam', loss='mse')
    #Somehow train the network here to guess SNR
    train_model.fit_generator(generator, epochs=epochs)
    incp_model.trainable = False
    return

class DataGenerator(keras.utils.Sequence):
    def __init__(self, signals, noise, signal_labels, index_list,
                 noise_snr=4., batch_size=32, shuffle=True,
                 label_type='snr_and_psc', use_inputs=None):
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
        label_type : {str, 'snr_and_psc'}
            Which output the label should be. Must be either
            'snr_and_psc' (gets the SNR and p-score as output) or
            'signal' (gets the pure signal as output) or 'snr' (get just
            the SNR as output).
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
        if label_type in ['snr_and_psc', 'signals', 'snr']:
            self.label_type = label_type
        else:
            raise ValueError('Unknown label_type {}.'.format(label_type))
        self.use_inputs = list(range(8)) if use_inputs is None else use_inputs
        self.data_channels = 8
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
    #The number of epochs every part will be trained.
    epochs_per_part = 20
    
    net_path = os.path.join(get_store_path(), 'transfer_learning_test')
    if not os.path.isdir(net_path):
        os.mkdir(net_path)
    
    #Set use_local to True for testing on the local laptop.
    use_local = False
    if use_local:
        file_path = os.path.join(get_store_path(), 'templates_more_right.hf5')
    else:
        file_path = os.path.join(get_store_path(), 'templates_more_right_vary_all_final.hf5')
    with h5py.File(file_path, 'r') as FILE:
        if use_local:
            signals = FILE['signals']['data'][:7]
            noise = FILE['noise']['data'][:7]
            signal_labels = FILE['signals']['snr'][:7]
        else:
            signals = FILE['signals']['data'][:2000]
            noise = FILE['noise']['data'][:10000]
            signal_labels = FILE['signals']['snr'][:2000]
    
    if use_local:
        noise_index_list = generate_unique_index_pairs_noise(noise, 7)
        signal_index_list = generate_unique_index_pairs(signals, noise, 7)
    else:
        noise_index_list = generate_unique_index_pairs_noise(noise, 10000)
        signal_index_list = generate_unique_index_pairs(signals, noise, 10000)
    index_list = noise_index_list + signal_index_list
    np.random.shuffle(index_list)
    
    NUM_DETECTORS = 2
    print("Training TCN model 1")
    tcn_generator = DataGenerator(signals, noise, signal_labels, index_list, label_type='signals', use_inputs=[0])
    inp1, is1, in1 = get_input(('Input_signal_1', 'Input_noise_1'), NUM_OF_DETECTORS=NUM_DETECTORS)
    tcn1 = get_tcn_stack(inp1, name="TCN_out_1s_part_1")
    tcn_model_1 = keras.models.Model(inputs=[is1, in1], outputs=[tcn1])
    train_tcn(tcn_model_1, tcn_generator, epochs=epochs_per_part)
    #print("Done training the following model:")
    #print(tcn_model_1.summary())
    
    print("Training TCN model 2")
    tcn_generator = DataGenerator(signals, noise, signal_labels, index_list, label_type='signals', use_inputs=[1])
    #print("Got generator")
    inp2, is2, in2 = get_input(('Input_signal_2', 'Input_noise_2'), NUM_OF_DETECTORS=NUM_DETECTORS)
    #print("Generated input layers")
    tcn2 = get_tcn_stack(inp2, name="TCN_out_1s_part_2")
    #print("Generated TCN")
    tcn_model_2 = keras.models.Model(inputs=[is2, in2], outputs=[tcn2])
    #print("Generated model. Start training TCN model 2 now.")
    train_tcn(tcn_model_2, tcn_generator, epochs=epochs_per_part)
    #print("Done training the following model:")
    #print(tcn_model_2.summary())
    
    print("Training TCN model 3")
    tcn_generator = DataGenerator(signals, noise, signal_labels, index_list, label_type='signals', use_inputs=[2])
    inp3, is3, in3 = get_input(('Input_signal_3', 'Input_noise_3'), NUM_OF_DETECTORS=NUM_DETECTORS)
    tcn3 = get_tcn_stack(inp3, name="TCN_out_2s")
    tcn_model_3 = keras.models.Model(inputs=[is3, in3], outputs=[tcn3])
    train_tcn(tcn_model_3, tcn_generator, epochs=epochs_per_part)
    
    print("Training TCN model 4")
    tcn_generator = DataGenerator(signals, noise, signal_labels, index_list, label_type='signals', use_inputs=[3])
    inp4, is4, in4 = get_input(('Input_signal_4', 'Input_noise_4'), NUM_OF_DETECTORS=NUM_DETECTORS)
    tcn4 = get_tcn_stack(inp4, name="TCN_out_4s")
    tcn_model_4 = keras.models.Model(inputs=[is4, in4], outputs=[tcn4])
    train_tcn(tcn_model_4, tcn_generator, epochs=epochs_per_part)
    
    print("Training TCN model 5")
    tcn_generator = DataGenerator(signals, noise, signal_labels, index_list, label_type='signals', use_inputs=[4])
    inp5, is5, in5 = get_input(('Input_signal_5', 'Input_noise_5'), NUM_OF_DETECTORS=NUM_DETECTORS)
    tcn5 = get_tcn_stack(inp5, name="TCN_out_8s")
    tcn_model_5 = keras.models.Model(inputs=[is5, in5], outputs=[tcn5])
    train_tcn(tcn_model_5, tcn_generator, epochs=epochs_per_part)
    
    print("Training TCN model 6")
    tcn_generator = DataGenerator(signals, noise, signal_labels, index_list, label_type='signals', use_inputs=[5])
    inp6, is6, in6 = get_input(('Input_signal_6', 'Input_noise_6'), NUM_OF_DETECTORS=NUM_DETECTORS)
    tcn6 = get_tcn_stack(inp6, name="TCN_out_16s")
    tcn_model_6 = keras.models.Model(inputs=[is6, in6], outputs=[tcn6])
    train_tcn(tcn_model_6, tcn_generator, epochs=epochs_per_part)
    
    print("Training TCN model 7")
    tcn_generator = DataGenerator(signals, noise, signal_labels, index_list, label_type='signals', use_inputs=[6])
    inp7, is7, in7 = get_input(('Input_signal_7', 'Input_noise_7'), NUM_OF_DETECTORS=NUM_DETECTORS)
    tcn7 = get_tcn_stack(inp7, name="TCN_out_32s")
    tcn_model_7 = keras.models.Model(inputs=[is7, in7], outputs=[tcn7])
    train_tcn(tcn_model_7, tcn_generator, epochs=epochs_per_part)
    
    print("Training TCN model 8")
    tcn_generator = DataGenerator(signals, noise, signal_labels, index_list, label_type='signals', use_inputs=[7])
    inp8, is8, in8 = get_input(('Input_signal_8', 'Input_noise_8'), NUM_OF_DETECTORS=NUM_DETECTORS)
    tcn8 = get_tcn_stack(inp8, name="TCN_out_64s")
    tcn_model_8 = keras.models.Model(inputs=[is8, in8], outputs=[tcn8])
    train_tcn(tcn_model_8, tcn_generator, epochs=epochs_per_part)
    
    print("Training inception model 1")
    inc_generator = DataGenerator(signals, noise, signal_labels, index_list, label_type='snr', use_inputs=[0])
    pre1 = preprocess(tcn_model_1([is1, in1]))
    inc_block_1 = inception_block(pre1)
    model_inc_block_1 = keras.models.Model(inputs=[is1, in1], outputs=inc_block_1)
    train_inception_part(model_inc_block_1, inc_generator, epochs=epochs_per_part, inputs=[is1, in1], name='Inception_Block_1')
    
    print("Training inception model 2")
    inc_generator = DataGenerator(signals, noise, signal_labels, index_list, label_type='snr', use_inputs=[1])
    pre2 = preprocess(tcn_model_2([is2, in2]))
    inc_block_2 = inception_block(pre2)
    model_inc_block_2 = keras.models.Model(inputs=[is2, in2], outputs=inc_block_2)
    train_inception_part(model_inc_block_2, inc_generator, epochs=epochs_per_part, inputs=[is2, in2], name='Inception_Block_2')
    
    print("Training inception model 3")
    inc_generator = DataGenerator(signals, noise, signal_labels, index_list, label_type='snr', use_inputs=[2])
    pre3 = preprocess(tcn_model_3([is3, in3]))
    inc_block_3 = inception_block(pre3)
    model_inc_block_3 = keras.models.Model(inputs=[is3, in3], outputs=inc_block_3)
    train_inception_part(model_inc_block_3, inc_generator, epochs=epochs_per_part, inputs=[is3, in3], name='Inception_Block_3')
    
    print("Training inception model 4")
    inc_generator = DataGenerator(signals, noise, signal_labels, index_list, label_type='snr', use_inputs=[3])
    pre4 = preprocess(tcn_model_4([is4, in4]))
    inc_block_4 = inception_block(pre4)
    model_inc_block_4 = keras.models.Model(inputs=[is4, in4], outputs=inc_block_4)
    train_inception_part(model_inc_block_4, inc_generator, epochs=epochs_per_part, inputs=[is4, in4], name='Inception_Block_4')
    
    print("Training inception model 5")
    inc_generator = DataGenerator(signals, noise, signal_labels, index_list, label_type='snr', use_inputs=[4])
    pre5 = preprocess(tcn_model_5([is5, in5]))
    inc_block_5 = inception_block(pre5)
    model_inc_block_5 = keras.models.Model(inputs=[is5, in5], outputs=inc_block_5)
    train_inception_part(model_inc_block_5, inc_generator, epochs=epochs_per_part, inputs=[is5, in5], name='Inception_Block_5')
    
    print("Training inception model 6")
    inc_generator = DataGenerator(signals, noise, signal_labels, index_list, label_type='snr', use_inputs=[5])
    pre6 = preprocess(tcn_model_6([is6, in6]))
    inc_block_6 = inception_block(pre6)
    model_inc_block_6 = keras.models.Model(inputs=[is6, in6], outputs=inc_block_6)
    train_inception_part(model_inc_block_6, inc_generator, epochs=epochs_per_part, inputs=[is6, in6], name='Inception_Block_6')
    
    print("Training inception model 7")
    inc_generator = DataGenerator(signals, noise, signal_labels, index_list, label_type='snr', use_inputs=[6])
    pre7 = preprocess(tcn_model_7([is7, in7]))
    inc_block_7 = inception_block(pre7)
    model_inc_block_7 = keras.models.Model(inputs=[is7, in7], outputs=inc_block_7)
    train_inception_part(model_inc_block_7, inc_generator, epochs=epochs_per_part, inputs=[is7, in7], name='Inception_Block_7')
    
    print("Training inception model 8")
    inc_generator = DataGenerator(signals, noise, signal_labels, index_list, label_type='snr', use_inputs=[7])
    pre8 = preprocess(tcn_model_8([is8, in8]))
    inc_block_8 = inception_block(pre8)
    model_inc_block_8 = keras.models.Model(inputs=[is8, in8], outputs=inc_block_8)
    train_inception_part(model_inc_block_8, inc_generator, epochs=epochs_per_part, inputs=[is8, in8], name='Inception_Block_8')
    
    print("Training inception model 9")
    inc_generator = DataGenerator(signals, noise, signal_labels, index_list, label_type='snr', use_inputs=[0, 1])
    concat1 = keras.layers.concatenate([model_inc_block_1([is1, in1]), model_inc_block_2([is2, in2])])
    inc_block_9 = inception_block(concat1)
    model_inc_block_9 = keras.models.Model(inputs=[is1, in1, is2, in2], outputs=inc_block_9)
    train_inception_part(model_inc_block_9, inc_generator, epochs=epochs_per_part, inputs=[is1, in1, is2, in2], name='Inception_Block_9')
    
    print("Training inception model 10")
    inc_generator = DataGenerator(signals, noise, signal_labels, index_list, label_type='snr', use_inputs=[2, 3])
    concat2 = keras.layers.concatenate([model_inc_block_3([is3, in3]), model_inc_block_4([is4, in4])])
    inc_block_10 = inception_block(concat2)
    model_inc_block_10 = keras.models.Model(inputs=[is3, in3, is4, in4], outputs=inc_block_10)
    train_inception_part(model_inc_block_10, inc_generator, epochs=epochs_per_part, inputs=[is3, in3, is4, in4], name='Inception_Block_10')
    
    print("Training inception model 11")
    inc_generator = DataGenerator(signals, noise, signal_labels, index_list, label_type='snr', use_inputs=[4, 5])
    concat3 = keras.layers.concatenate([model_inc_block_5([is5, in5]), model_inc_block_6([in6, is6])])
    inc_block_11 = inception_block(concat3)
    model_inc_block_11 = keras.models.Model(inputs=[is5, in5, is6, in6], outputs=inc_block_11)
    train_inception_part(model_inc_block_11, inc_generator, epochs=epochs_per_part, inputs=[is5, in5, is6, in6], name='Inception_Block_11')
    
    print("Training inception model 12")
    inc_generator = DataGenerator(signals, noise, signal_labels, index_list, label_type='snr', use_inputs=[6, 7])
    concat4 = keras.layers.concatenate([model_inc_block_7([is7, in7]), model_inc_block_8([is8, in8])])
    inc_block_12 = inception_block(concat4)
    model_inc_block_12 = keras.models.Model(inputs=[is7, in7, is8, in8], outputs=inc_block_12)
    train_inception_part(model_inc_block_12, inc_generator, epochs=epochs_per_part, inputs=[is7, in7, is8, in8], name='Inception_Block_12')
    
    print("Training inception model 13")
    inc_generator = DataGenerator(signals, noise, signal_labels, index_list, label_type='snr', use_inputs=[0, 1, 2, 3])
    concat5 = keras.layers.concatenate([model_inc_block_9([is1, in1, is2, in2]), model_inc_block_10([is3, in3, is4, in4])])
    inc_block_13 = inception_block(concat5)
    model_inc_block_13 = keras.models.Model(inputs=[is1, in1, is2, in2, is3, in3, is4, in4], outputs=inc_block_13)
    train_inception_part(model_inc_block_13, inc_generator, epochs=epochs_per_part, inputs=[is1, in1, is2, in2, is3, in3, is4, in4], name='Inception_Block_13')
    
    print("Training inception model 14")
    inc_generator = DataGenerator(signals, noise, signal_labels, index_list, label_type='snr', use_inputs=[4, 5, 6, 7])
    concat6 = keras.layers.concatenate([model_inc_block_11([is5, in5, is6, in6]), model_inc_block_12([is7, in7, is8, in8])])
    inc_block_14 = inception_block(concat6)
    model_inc_block_14 = keras.models.Model(inputs=[is5, in5, is6, in6, is7, in7, is8, in8], outputs=inc_block_14)
    train_inception_part(model_inc_block_14, inc_generator, epochs=epochs_per_part, inputs=[is5, in5, is6, in6, is7, in7, is8, in8], name='Inception_Block_14')
    
    with h5py.File(file_path, 'r') as FILE:
        if use_local:
            val_signals = FILE['signals']['data'][7:]
            val_noise = FILE['noise']['data'][7:]
            val_signal_labels = FILE['signals']['snr'][7:]
        else:
            val_signals = FILE['signals']['data'][2000:3000]
            val_noise = FILE['noise']['data'][10000:15000]
            val_signal_labels = FILE['signals']['snr'][2000:3000]
    if use_local:
        val_noise_index_list = generate_unique_index_pairs_noise(val_noise, 3)
        val_signal_index_list = generate_unique_index_pairs(val_signals, val_noise, 3)
    else:
        val_noise_index_list = generate_unique_index_pairs_noise(val_noise, 5000)
        val_signal_index_list = generate_unique_index_pairs(val_signals, val_noise, 10000)
    val_index_list = val_noise_index_list + val_signal_index_list
    np.random.shuffle(val_index_list)
    val_generator = DataGenerator(val_signals, val_noise, val_signal_labels, val_index_list, label_type='snr_and_psc', shuffle=False)
    
    print("Training final model")
    inc_generator = DataGenerator(signals, noise, signal_labels, index_list, label_type='snr_and_psc')
    concat7 = keras.layers.concatenate([model_inc_block_13([is1, in1, is2, in2, is3, in3, is4, in4]), model_inc_block_14([is5, in5, is6, in6, is7, in7, is8, in8])])
    pool_2 = keras.layers.MaxPooling1D(4)(concat7)
    dim_red = keras.layers.Conv1D(32, 1)(pool_2)
    flatten = keras.layers.Flatten()(dim_red)
    
    dense_1 = keras.layers.Dense(2)(flatten)
    dense_2 = keras.layers.Dense(1, activation='relu', name='Out_SNR')(dense_1)
    
    dense_3 = keras.layers.Dense(3)(flatten)
    dense_4 = keras.layers.Dense(2, activation='softmax', name='Out_Bool')(dense_3)
    
    final_model = keras.models.Model(inputs=[is1, in1, is2, in2, is3, in3, is4, in4, is5, in5, is6, in6, is7, in7, is8, in8], outputs=[dense_2, dense_4])
    final_model.compile(optimizer='adam', loss=['mse', 'categorical_crossentropy'], loss_weights=[1.0, 0.5], metrics=['mape', 'accuracy'])
    
    SensTracker = cc.SensitivityTracker(val_generator, net_path, interval=1)
    check_path = os.path.join(net_path, 'transfer_learning_test_epoch_{epoch:d}.hf5')
    ModelCheck = ModelCheckpoint(check_path, verbose=1, period=1)
    
    final_model.fit_generator(inc_generator, epochs=epochs_per_part, validation_data=val_generator, callbacks=[ModelCheck, SensTracker])
    final_model.save(os.path.join(net_path, 'final_epoch.hf5'))
    print("Trained the complete model as given in:")
    print(final_model.summary())
    print(final_model.evaluate_generator(val_generator))
    return

def train_model(model, dobj, net_path, epochs=None, epoch_break=10, batch_size=32):
    print("Epochs: {}\nEpoch_break={}".format(epochs, epoch_break))
    print("Net path: {}".format(net_path))
    name = os.path.basename(__file__)[:-4]
    
    #Store the results of training (i.e. the loss)
    results = []
    
    #Check if epochs is None, if so try to train until the loss of the trainingsset and the one of the testingset seperate by too much
    if epochs == None:
        keepRunning = True
        curr_counter = 0
        
        #Keep training for the number of epochs specified in epoch_break
        while keepRunning:
            #Fit data to model
            model.fit(train_data, train_labels, epochs=epoch_break)
            
            curr_counter += epoch_break
            
            #Save after every training-cycle
            model.save(os.path.join(net_path, name + "_epoch_" + str(curr_counter) + ".hf5"))
            
            #Evaluate the net and store the values
            results.append([curr_counter, model.evaluate(train_data, train_labels), model.evaluate(test_data, test_labels)])
            
            #Train at least 5 times, after that keep training only if no overfitting happens
            if len(results) >= 5:
                start_index = int(curr_counter / epoch_break) - 1
                train_loss = [dat[1][0] for dat in results[start_index:start_index+5]]
                test_loss = [dat[2][0] for dat in results[start_index:start_index+5]]
                keepRunning = not evaluate_overfitting(train_loss, test_loss)
                
    else:
        #Check if epochs are a smiple multiple of epoch_break, meaning that it should train for an integer number of cycles
        if epochs % epoch_break == 0:
            ran = int(epochs / epoch_break)
        #If not, train one more cycle and train only for the left amount of epochs in the last cycle.
        else:
            ran = int(epochs / epoch_break) + 1
        
        #Count how many epochs have passed
        curr_counter = 0
        
        print("Expected memory_size: {}".format(get_model_memory_usage(batch_size, model)))
        
        training_generator = get_generator()
        testing_generator = get_generator()
        
        training_generator = training_generator(dobj.loaded_train_data, dobj.loaded_train_labels, batch_size=batch_size)
        testing_generator = testing_generator(dobj.loaded_test_data, dobj.loaded_test_labels, batch_size=batch_size)
        
        q_size = 2
        SensTracker = cc.SensitivityTracker(testing_generator, net_path, interval=epoch_break)
        check_path = os.path.join(net_path, name + '_epoch_{epoch:d}.hf5')
        ModelCheck = ModelCheckpoint(check_path, verbose=1, period=epoch_break)
        
        tmp = model.fit_generator(generator=training_generator, validation_data=testing_generator, epochs=epochs, max_q_size=q_size, callbacks=[ModelCheck, SensTracker]).history
        
        results = []
        keys = tmp.keys()
        for i, ep in enumerate(np.arange(epoch_break, epochs + epoch_break, epoch_break, dtype=int)):
            tr_metrics = []
            te_metrics = []
            for k in keys:
                if k == 'loss':
                    tr_metrics.insert(0, tmp[k][i])
                elif k == 'val_loss':
                    te_metrics.insert(0, tmp[k][i])
                elif 'val' in k:
                    te_metrics.append(tmp[k][i])
                else:
                    tr_metrics.append(tmp[k][i])
            results.append([ep, tr_metrics, te_metrics])
    
    #Save the results to a file.
    with open(os.path.join(net_path, name + '_results.json'), "w+") as FILE:
        json.dump(results, FILE, indent=4)
    
    with open(os.path.join(net_path, name + '_results_keyed.json'), "w+") as FILE:
        json.dump(tmp, FILE, indent=4)
    
    return(model)


if __name__ == "__main__":
    main()
