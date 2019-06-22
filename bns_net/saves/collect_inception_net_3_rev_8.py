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

def incp_lay(x, filter_num):
    active_filter_sizes = (4, 8, 16)
    l = keras.layers.Conv1D(3 * filter_num, active_filter_sizes[0], padding='same', activation='relu')(x)
    lm_1 = keras.layers.Conv1D(int(round(filter_num / 2.0)), 1, activation='relu')(x)
    lm_2 = keras.layers.Conv1D(2 * filter_num, active_filter_sizes[1], padding='same', activation='relu')(lm_1)
    rm_1 = keras.layers.Conv1D(int(round(filter_num / 2.0)), 1, activation='relu')(x)
    rm_2 = keras.layers.Conv1D(filter_num, active_filter_sizes[2], padding='same', activation='relu')(rm_1)
    r_1 = keras.layers.MaxPooling1D(4, strides=1, padding='same')(x)
    r_2 = keras.layers.Conv1D(int(round(filter_num)), 1, activation='relu')(r_1)
    
    outp = keras.layers.concatenate([l, lm_2, rm_2, r_2])
    
    return(outp)

def stack(x, NUM_DETECTORS, DROPOUT_RATE):
    batch_1 = keras.layers.BatchNormalization()(x)
    conv_1 = keras.layers.Conv1D(64, 32)(batch_1)
    bn_conv_1 = keras.layers.BatchNormalization()(conv_1)
    act_conv_1 = keras.layers.Activation('relu')(bn_conv_1)
    pool_conv_1 = keras.layers.MaxPooling1D(4)(act_conv_1)
    conv_2 = keras.layers.Conv1D(128, 16)(pool_conv_1)
    bn_conv_2 = keras.layers.BatchNormalization()(conv_2)
    act_conv_2 = keras.layers.Activation('relu')(bn_conv_2)
    inc_1 = incp_lay(act_conv_2, 32)
    batch_2 = keras.layers.BatchNormalization()(inc_1)
    inc_2 = incp_lay(batch_2, 32)
    pool_1 = keras.layers.MaxPooling1D(2)(inc_2)
    batch_3 = keras.layers.BatchNormalization()(pool_1)
    inc_3 = incp_lay(batch_3, 32)
    batch_4 = keras.layers.BatchNormalization()(inc_3)
    return(batch_4)

def get_input(input_names, NUM_OF_DETECTORS=2):
    inp_sig = keras.layers.Input(shape=(4096, NUM_OF_DETECTORS), name=input_names[0])
    inp_noi = keras.layers.Input(shape=(4096, NUM_OF_DETECTORS), name=input_names[1])
    add = keras.layers.Add()([inp_sig, inp_noi])
    return((add, inp_sig, inp_noi))

def get_model():
    NUM_DETECTORS = 2
    DROPOUT_RATE = 0.25
    
    inp2, is2, in2 = get_input(("Signal_2s", "Noise_2s"))
    inp8, is8, in8 = get_input(("Signal_8s", "Noise_8s"))
    inp32, is32, in32 = get_input(("Signal_32s", "Noise_32s"))
    
    stack_2s = stack(inp2, NUM_DETECTORS, DROPOUT_RATE)
    stack_8s = stack(inp8, NUM_DETECTORS, DROPOUT_RATE)
    stack_32s = stack(inp32, NUM_DETECTORS, DROPOUT_RATE)
    
    #COMBINED OUTPUT
    combined = keras.layers.concatenate([stack_2s, stack_8s, stack_32s])
    
    inc_4 = incp_lay(combined, 32)
    batch_4 = keras.layers.BatchNormalization()(inc_4)
    pool_2 = keras.layers.MaxPooling1D(4)(batch_4)
    inc_5 = incp_lay(pool_2, 32)
    batch_5 = keras.layers.BatchNormalization()(inc_5)
    dim_red = keras.layers.Conv1D(16, 1)(batch_5)
    
    #FLAT LAYERS
    flatten = keras.layers.Flatten()(dim_red)
    
    dense_1 = keras.layers.Dense(2)(flatten)
    dense_2 = keras.layers.Dense(1, activation='relu', name='Out_SNR')(dense_1)
    
    dense_3 = keras.layers.Dense(3)(flatten)
    dense_4 = keras.layers.Dense(2, activation='softmax', name='Out_Bool')(dense_3)
    
    model = keras.models.Model(inputs=[is2, in2, is8, in8, is32, in32], outputs=[dense_2, dense_4])
    
    return(model)

def compile_model(model):
    model.compile(loss=['mean_squared_error', 'categorical_crossentropy'], loss_weights=[1.0, 0.5], optimizer='adam', metrics=['mape', 'accuracy'])

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
                self.signals = FILE['signals']['data'][:]
                self.noise = FILE['noise']['data'][:]
                self.signal_labels = FILE['signals']['snr'][:]
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
            
            signal_indices = self.generate_unique_index_pairs(2*num_signals)
            noise_indices = list(np.arange(0, len(self.noise), dtype=int))
            sig_split = int(float(len(self.signals)) * 3 / 4)
            self.training_indices = np.array(self.generate_unique_index_pairs(num_signals, signal_index_range=[0, signal_split], noise_index_range=[0, num_noise]) + noise_indices[:num_noise])
            np.random.shuffle(self.training_indices)
            self.training_indices = [(pt[0], pt[1]) for pt in self.training_indices]
            self.testing_indices = np.array(signal_indices[num_signals:] + noise_indices[num_noise:])
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
            elif (not isinstance(noise_index_range) or not isinstance(noise_index_range)) and len(noise_index_range) == 2:
                raise ValueError('noise_index_range needs to be a list or tuple of length 2.')
            
            if signal_index_range == None:
                signal_index_range = [0, len(self.signals)]
            elif (not isinstance(signal_index_range) or not isinstance(signal_index_range)) and len(signal_index_range) == 2:
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
        self.data_channels = 3
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
        X = [np.zeros([len(indices), 2, self.signals[0].shape[0]]) for i in range(self.data_channels * 2)]
        
        y_1 = np.zeros((len(indices), 1))
        
        y_2 = np.zeros((len(indices), 2))
        
        clear_list = [1, 3, 5]
        
        for i, idx in enumerate(indices):
            sig_ind, noi_ind = self.index_list[idx]
            
            for j in range(self.data_channels):
                ci = clear_list[j]
                if not sig_ind == -1:
                    X[2 * j][i][0] = self.signals[sig_ind].transpose()[ci]
                    X[2 * j][i][1] = self.signals[sig_ind].transpose()[ci+7]
                
                X[2 * j + 1][i][0] = self.noise[noi_ind].transpose()[ci]
                X[2 * j + 1][i][1] = self.noise[noi_ind].transpose()[ci]
            
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
