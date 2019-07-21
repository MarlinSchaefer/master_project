import keras
import numpy as np
import json
import os
import load_data
import generator as g
from data_object import DataSet
from aux_functions import get_store_path
import h5py
from evaluate_nets import evaluate_training
import time
from keras import backend as K
from custom_layers import custom_loss, loss_c1

filter_size = (1, 2, 3)

#def custom_loss(y_true, y_pred):
    #part1  = 4 * (y_true - y_pred) / np.e
    #part1 *= (1 - 2 * K.cast(y_true > 6, K.floatx()))
    #part1 -= 1
    #part1 *= K.cast(y_true - y_pred < -1, K.floatx())
    
    #part21 = 4 * K.exp(y_pred - y_true + 2) / (np.e ** 2 * K.square(y_pred - y_true + 2))
    #part22 = 4 * K.exp(y_true - y_pred + 2) / (np.e ** 2 * K.square(y_true - y_pred + 2))
    #part2  = K.cast(y_true <= 6, K.floatx()) * part21 + K.cast(y_true > 6, K.floatx()) * part22
    #part2 *= K.cast(y_true - y_pred >= -1, K.floatx())
    
    #return K.mean(K.minimum(part1 + part2, K.abs(y_true - y_pred) + 10000))

def set_filter_size(tup):
    global filter_size
    if isinstance(tup, tuple) and len(tup) == 3:
        filter_size = tup

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
    #ATTENTION: Drop this maybe
    drop_1 = keras.layers.Dropout(DROPOUT_RATE)(batch_1)
    conv_1 = keras.layers.Conv1D(64, 32)(drop_1)
    bn_conv_1 = keras.layers.BatchNormalization()(conv_1)
    act_conv_1 = keras.layers.Activation('relu')(bn_conv_1)
    pool_conv_1 = keras.layers.MaxPooling1D(4)(act_conv_1)
    conv_2 = keras.layers.Conv1D(128, 16)(pool_conv_1)
    bn_conv_2 = keras.layers.BatchNormalization()(conv_2)
    act_conv_2 = keras.layers.Activation('relu')(bn_conv_2)
    return(act_conv_2)

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
    
    #Maybe this preprocessing can be reduced and one just needs to add
    #BatchNormalization and Dropout, but no convolution layers.
    lvl_1_1 = preprocess(inp1)
    lvl_1_2 = preprocess(inp2)
    lvl_1_3 = preprocess(inp3)
    lvl_1_4 = preprocess(inp4)
    lvl_1_5 = preprocess(inp5)
    lvl_1_6 = preprocess(inp6)
    lvl_1_7 = preprocess(inp7)
    lvl_1_8 = preprocess(inp8)
    
    lvl_2_1 = combine_stack(lvl_1_1, lvl_1_2)
    lvl_2_2 = combine_stack(lvl_1_3, lvl_1_4)
    lvl_2_3 = combine_stack(lvl_1_5, lvl_1_6)
    lvl_2_4 = combine_stack(lvl_1_7, lvl_1_8)
    
    lvl_3_1 = combine_stack(lvl_2_1, lvl_2_2)
    lvl_3_2 = combine_stack(lvl_2_3, lvl_2_4)
    
    lvl_4_1 = combine_stack(lvl_3_1, lvl_3_2)
    
    pool_2 = keras.layers.MaxPooling1D(4)(lvl_4_1)
    dim_red = keras.layers.Conv1D(32, 1)(pool_2)
    flatten = keras.layers.Flatten()(dim_red)
    
    dense_1 = keras.layers.Dense(2)(flatten)
    dense_2 = keras.layers.Dense(1, name='Out_SNR')(dense_1)
    
    dense_3 = keras.layers.Dense(3)(flatten)
    dense_4 = keras.layers.Dense(2, activation='softmax', name='Out_Bool')(dense_3)
    
    model = keras.models.Model(inputs=[is1, in1, is2, in2, is3, in3, is4, in4, is5, in5, is6, in6, is7, in7, is8, in8], outputs=[dense_2, dense_4])
    
    return(model)

def compile_model(model):
    #opt = keras.optimizers.Adam(lr=10**-6)
    model.compile(loss=[loss_c1, 'categorical_crossentropy'], loss_weights=[1.0, 0.5], optimizer='adam', metrics={'Out_SNR': 'mape', 'Out_Bool': 'accuracy'})

def evaluate_overfitting(train_loss, test_loss):
    THRESHOLD = 0.7
    percentage_loss_difference = [abs(train_loss[i] - test_loss[i]) / train_loss[i] for i in range(len(train_loss))]
    
    bigger_then_threshold_q = [bool(pt > THRESHOLD) for pt in percentage_loss_difference]
    
    if False in bigger_then_threshold_q:
        return(False)
    else:
        return_true = 0
        
        for i in range(len(bigger_then_threshold_q)):
            if return_true == i:
                if i == 0:
                    return_true += 1
                elif test_loss[i] < test_loss[i-1]:
                    return_true += 1
            else:
                return(False)
    
    return(True)

def get_data_obj(file_path):
    #This is a real dirty hack to the data_object.DataSet. It does
    #require a very special generator to work in the run_net framework.
    class CustomDataSet():
        def __init__(self, file_path):
            self.file_path = file_path
            self.load_data()
        
        def load_data(self):
            with h5py.File(self.file_path) as FILE:
                self.signals = FILE['signals']['data'][:10000]
                self.noise = FILE['noise']['data'][:50000]
                self.signal_labels = FILE['signals']['snr'][:10000]
                self.noise_label = FILE['noise']['snr'][0]
        
        @property
        def loaded_train_data(self):
            return((self.signals, self.noise, self.signal_labels, self.training_indices, self.noise_label))
        
        @property
        def loaded_test_data(self):
            return((self.signals, self.noise, self.signal_labels, self.testing_indices, self.noise_label))
        
        @property
        def loaded_test_labels(self):
            return(self.testing_labels)
        
        @property
        def loaded_train_labels(self):
            return(self.training_labels)
        
        @property
        def loaded_train_snr(self):
            return(np.zeros(len(self.training_indices)))
        
        @property
        def loaded_test_snr(self):
            return(np.zeros(len(self.testing_indices)))
        
        def get_set(self, slice=None):
            if slice == None:
                #num_pairs = len(self.signals) * len(self.noise) / 2
                num_noise = int(float(len(self.noise)) * 3 / 4)
                num_signals = num_noise
            elif type(slice) in [tuple, list] and len(slice) == 2:
                num_signals = slice[0]
                num_noise = slice[1]
            else:
                raise ValueError('slice needs to be a tuple or list of exactly 2 items.')
            
            noise_indices = np.arange(0, len(self.noise), dtype=int)
            noise_indices = [(-1, pt) for pt in noise_indices]
            signal_split = int(float(len(self.signals)) * 3 / 4)
            self.training_indices = np.array(self.generate_unique_index_pairs(num_signals, signal_index_range=[0, signal_split], noise_index_range=[0, num_noise]) + noise_indices[:num_noise])
            np.random.shuffle(self.training_indices)
            self.training_indices = [(pt[0], pt[1]) for pt in self.training_indices]
            self.testing_indices = np.array(self.generate_unique_index_pairs(num_signals, signal_index_range=[signal_split, len(self.signals)], noise_index_range=[num_noise, len(self.noise)]) + noise_indices[num_noise:])
            np.random.shuffle(self.testing_indices)
            self.testing_indices = [(pt[0], pt[1]) for pt in self.testing_indices]
            
            training_snrs = np.array([[self.signal_labels[i[0]]] if not i[0] == -1 else [self.noise_label] for i in self.training_indices])
            training_bool = np.array([[1.0, 0.0] if not i[0] == -1 else [0.0, 1.0] for i in self.training_indices])
            
            testing_snrs = np.array([[self.signal_labels[i[0]]] if not i[0] == -1 else [self.noise_label] for i in self.testing_indices])
            testing_bool = np.array([[1.0, 0.0] if not i[0] == -1 else [0.0, 1.0] for i in self.testing_indices])
            
            self.training_labels = [training_snrs, training_bool]
            self.testing_labels = [testing_snrs, testing_bool]
        
        def get_file_properties(self):
            return({'snr': [8.0, 15.0]})
        
        def generate_unique_index_pairs_noise(self, num_pairs, noise=None):
            if noise == None:
                noise = self.noise
            
            if len(noise) < num_pairs:
                raise ValueError("Can't generate more indices for pure noise than there are noise instances.")
            
            if num_pairs < len(noise) / 2:
                invert = False
            else:
                invert = True
                num_pairs = len(noise) - num_pairs
            
            curr_pairs = 0
            poss = np.zeros(len(noise))
            while curr_pairs < num_pairs:
                r_int = np.random.randint(0, len(noise))
                if poss[r_int] == 0:
                    poss[r_int] = 1
                    curr_pairs += 1
            
            true_val = 0 if invert else 1
            
            ret = []
            
            for i, val in enumerate(poss):
                if val == true_val:
                    ret.append((-1, i))
            
            ret = np.array(ret, dtype=int)
            np.random.shuffle(ret)
            
            ret = [(pt[0], pt[1]) for pt in ret]
            
            return(ret)
        
        def generate_unique_index_pairs(self, num_pairs, generate_signals_only=True, noise_index_range=None, signal_index_range=None):
            if noise_index_range == None:
                noise_index_range = [0, len(self.noise)]
            elif not isinstance(noise_index_range, list) and not isinstance(noise_index_range, tuple) and len(noise_index_range) == 2:
                raise ValueError('noise_index_range needs to be a list or tuple of length 2.')
            
            if signal_index_range == None:
                signal_index_range = [0, len(self.signals)]
            elif not isinstance(signal_index_range, list) and not isinstance(signal_index_range, tuple) and len(signal_index_range) == 2:
                raise ValueError('signal_index_range needs to be a list or tuple of length 2.')
            
            len_noi = noise_index_range[1] - noise_index_range[0]
            len_sig = signal_index_range[1] - signal_index_range[0]
            if len_sig < 0 or len_noi < 0 or noise_index_range[0] < 0 or signal_index_range[0] < 0 or noise_index_range[1] > len(self.noise) or signal_index_range[1] > len(self.signals):
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
        
        #The definitions below might not be sensible
        @property
        def training_data_shape(self):
            with h5py.File(self.file_path, 'r') as FILE:
                return(FILE['signals']['data'].shape)
        
        @property
        def training_label_shape(self):
            with h5py.File(self.file_path, 'r') as FILE:
                return(FILE['signals']['snr'].shape)
        
        @property
        def testing_data_shape(self):
            with h5py.File(self.file_path, 'r') as FILE:
                return(FILE['noise']['data'].shape)
            
        @property
        def training_label_shape(self):
            with h5py.File(self.file_path, 'r') as FILE:
                return(FILE['noise']['snr'].shape)
    
    return(CustomDataSet(file_path))

class DataGenerator(keras.utils.Sequence):
    def __init__(self, dobj_data, dobj_labels, batch_size=32, shuffle=True):
        (signals, noise, signal_labels, index_list, noise_snr) = dobj_data
        self.signals = signals
        self.noise = noise
        self.signal_labels = signal_labels
        self.index_list = index_list
        self.noise_snr = noise_snr
        self.batch_size = batch_size
        self.shuffle = shuffle
        #ATTENTION: Changed this to fit the three channels
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
        num_channels = self.data_channels
        num_inputs = 16
        X = [np.zeros([len(indices), num_detectors, 2048]) for i in range(num_inputs)]
        
        y_1 = np.zeros((len(indices), 1))
        
        y_2 = np.zeros((len(indices), 2))
        
        for i, idx in enumerate(indices):
            sig_ind, noi_ind = self.index_list[idx]
            
            if not sig_ind == -1:
                X[0][i][0] = self.signals[sig_ind].transpose()[0][2048:]
                X[0][i][1] = self.signals[sig_ind].transpose()[7][2048:]
                X[2][i][0] = self.signals[sig_ind].transpose()[0][:2048]
                X[2][i][1] = self.signals[sig_ind].transpose()[7][:2048]
                X[4][i][0] = self.signals[sig_ind].transpose()[1][:2048]
                X[4][i][1] = self.signals[sig_ind].transpose()[8][:2048]
                X[6][i][0] = self.signals[sig_ind].transpose()[2][:2048]
                X[6][i][1] = self.signals[sig_ind].transpose()[9][:2048]
                X[8][i][0] = self.signals[sig_ind].transpose()[3][:2048]
                X[8][i][1] = self.signals[sig_ind].transpose()[10][:2048]
                X[10][i][0] = self.signals[sig_ind].transpose()[4][:2048]
                X[10][i][1] = self.signals[sig_ind].transpose()[11][:2048]
                X[12][i][0] = self.signals[sig_ind].transpose()[5][:2048]
                X[12][i][1] = self.signals[sig_ind].transpose()[12][:2048]
                X[14][i][0] = self.signals[sig_ind].transpose()[6][:2048]
                X[14][i][1] = self.signals[sig_ind].transpose()[13][:2048]
            
            X[1][i][0] = self.noise[noi_ind].transpose()[0][2048:]
            X[1][i][1] = self.noise[noi_ind].transpose()[7][2048:]
            X[3][i][0] = self.noise[noi_ind].transpose()[0][:2048]
            X[3][i][1] = self.noise[noi_ind].transpose()[7][:2048]
            X[5][i][0] = self.noise[noi_ind].transpose()[1][:2048]
            X[5][i][1] = self.noise[noi_ind].transpose()[8][:2048]
            X[7][i][0] = self.noise[noi_ind].transpose()[2][:2048]
            X[7][i][1] = self.noise[noi_ind].transpose()[9][:2048]
            X[9][i][0] = self.noise[noi_ind].transpose()[3][:2048]
            X[9][i][1] = self.noise[noi_ind].transpose()[10][:2048]
            X[11][i][0] = self.noise[noi_ind].transpose()[4][:2048]
            X[11][i][1] = self.noise[noi_ind].transpose()[11][:2048]
            X[13][i][0] = self.noise[noi_ind].transpose()[5][:2048]
            X[13][i][1] = self.noise[noi_ind].transpose()[12][:2048]
            X[15][i][0] = self.noise[noi_ind].transpose()[6][:2048]
            X[15][i][1] = self.noise[noi_ind].transpose()[13][:2048]
            
            if not sig_ind == -1:
                y_1[i] = self.signal_labels[sig_ind]
                
                y_2[i][0] = 1.0
                y_2[i][1] = 0.0
            else:
                y_1[i] = self.noise_snr
                
                y_2[i][0] = 0.0
                y_2[i][1] = 1.0
            
        
        X = [dat.transpose(0, 2, 1) for dat in X]
        
        return((X, [y_1, y_2]))

def get_generator():
    return(DataGenerator)

def get_model_memory_usage(batch_size, model):
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

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
        
        for i in range(ran):
            print("ran: {}\ni: {}".format(ran, i))
            #If epochs were not an integer multiple of epoch_break, the last training cycle has to be smaller
            if i == int(epochs / epoch_break):
                epoch_break = epochs - (ran - 1) * epoch_break
                #Handle the exception of epochs < epoch_break
                if epoch_break < 0:
                    epoch_break += epoch_break
            
            q_size = 2
            
            #Fit data to model            
            model.fit_generator(generator=training_generator, epochs=epoch_break, max_q_size=q_size)
            
            #Iterate counter
            curr_counter += epoch_break
            print(curr_counter)
            
            #Store model after each training-cycle
            print("Net path before saving: {}".format(net_path))
            tmp_name = str(name + "_epoch_" + str(curr_counter) + ".hf5")
            tmp_path = os.path.join(net_path, tmp_name)
            print("Trying to save at: {}".format(tmp_path))
            model.save(os.path.join(net_path, tmp_name))
            print("Stored net at: {}".format(os.path.join(net_path, tmp_name)))
            
            #Evaluate the performance of the net after every cycle and store it.
            results.append([curr_counter, model.evaluate_generator(generator=training_generator, max_q_size=q_size), model.evaluate_generator(generator=testing_generator, max_q_size=q_size)])
            #print("Results: {}".format(results))
    
    #Save the results to a file.
    with open(os.path.join(net_path, name + '_results.json'), "w+") as FILE:
        json.dump(results, FILE, indent=4)
    
    return(model)
