import keras
import numpy as np
import h5py
import os
import json

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
    dropout_1 = keras.layers.Dropout(DROPOUT_RATE)(batch_1)
    conv_1 = keras.layers.Conv1D(64, 32)(dropout_1)
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

def get_model(num_detectors=2):
    NUM_DETECTORS = num_detectors
    DROPOUT_RATE = 0.25
    
    inp_1s_signal = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='Input_1s_signal')
    inp_2s_signal = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='Input_2s_signal')
    inp_4s_signal = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='Input_4s_signal')
    inp_8s_signal = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='Input_8s_signal')
    inp_16s_signal = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='Input_16s_signal')
    inp_32s_signal = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='Input_32s_signal')
    inp_64s_signal = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='Input_64s_signal')
    
    inp_1s_noise = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='Input_1s_noise')
    inp_2s_noise = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='Input_2s_noise')
    inp_4s_noise = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='Input_4s_noise')
    inp_8s_noise = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='Input_8s_noise')
    inp_16s_noise = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='Input_16s_noise')
    inp_32s_noise = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='Input_32s_noise')
    inp_64s_noise = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='Input_64s_noise')
    
    inp_1s = keras.layers.Add()([inp_1s_signal, inp_1s_noise])
    inp_2s = keras.layers.Add()([inp_2s_signal, inp_2s_noise])
    inp_4s = keras.layers.Add()([inp_4s_signal, inp_4s_noise])
    inp_8s = keras.layers.Add()([inp_8s_signal, inp_8s_noise])
    inp_16s = keras.layers.Add()([inp_16s_signal, inp_16s_noise])
    inp_32s = keras.layers.Add()([inp_32s_signal, inp_32s_noise])
    inp_64s = keras.layers.Add()([inp_64s_signal, inp_64s_noise])
    
    stack_1s = stack(inp_1s, NUM_DETECTORS, DROPOUT_RATE)
    stack_2s = stack(inp_2s, NUM_DETECTORS, DROPOUT_RATE)
    stack_4s = stack(inp_4s, NUM_DETECTORS, DROPOUT_RATE)
    stack_8s = stack(inp_8s, NUM_DETECTORS, DROPOUT_RATE)
    stack_16s = stack(inp_16s, NUM_DETECTORS, DROPOUT_RATE)
    stack_32s = stack(inp_32s, NUM_DETECTORS, DROPOUT_RATE)
    stack_64s = stack(inp_64s, NUM_DETECTORS, DROPOUT_RATE)
    
    #COMBINED OUTPUT
    combined = keras.layers.concatenate([stack_1s, stack_2s, stack_4s, stack_8s, stack_16s, stack_32s, stack_64s])
    
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
    
    model = keras.models.Model(inputs=[inp_1s_signal, inp_1s_noise, inp_2s_signal, inp_2s_noise, inp_4s_signal, inp_4s_noise, inp_8s_signal, inp_8s_noise, inp_16s_signal, inp_16s_noise, inp_32s_signal, inp_32s_noise, inp_64s_signal, inp_64s_noise], outputs=[dense_2, dense_4])
    
    return(model)

def compile_model(model):
    model.compile(loss={'SNR_out': 'mean_squared_error', 'Bool_out': 'categorical_crossentropy'}, loss_weights={'SNR_out': 1.0, 'Bool_out': 0.5}, optimizer='adam', metrics={'SNR_out': 'mape', 'Bool_out': 'accuracy'})

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
                num_pairs = len(self.signals) * len(self.noise) / 2
            elif type(slice) in [tuple, list] and len(slice) == 2:
                num_pairs = slice[0]
            else:
                raise ValueError('slice needs to be a tuple or list of exactly 2 items.')
            
            self.training_indices = self.generate_unique_index_pairs(num_pairs)
            self.testing_indices = self.generate_unique_index_pairs(int(float(num_pairs) * 3 / 7))
            
            training_snrs = np.array([[self.signal_labels[i[0]]] if not i[0] == -1 else [self.noise_label] for i in self.training_indices])
            training_bool = np.array([[1.0, 0.0] if not i[0] == -1 else [0.0, 1.0] for i in self.training_indices])
            
            testing_snrs = np.array([[self.signal_labels[i[0]]] if not i[0] == -1 else [self.noise_label] for i in self.testing_indices])
            testing_bool = np.array([[1.0, 0.0] if not i[0] == -1 else [0.0, 1.0] for i in self.testing_indices])
            
            self.training_labels = [training_snrs, training_bool]
            self.testing_labels = [testing_snrs, testing_bool]
        
        def get_file_properties(self):
            return({'snr': [8.0, 15.0]})
        
        def generate_unique_index_pairs(self, num_pairs):
            len_sig = len(self.signals)
            len_noi = len(self.noise)
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
            
            ret = [(pt[0], pt[1]) for pt in ret]
            
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
        self.data_channels = self.signals[0].shape[-1] / 2
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
        
        for i, idx in enumerate(indices):
            sig_ind, noi_ind = self.index_list[idx]
            
            for j in range(self.data_channels):
                if not sig_ind == -1:
                    X[2 * j][i][0] = self.signals[sig_ind].transpose()[j]
                    X[2 * j][i][1] = self.signals[sig_ind].transpose()[j+self.data_channels]
                
                X[2 * j + 1][i][0] = self.noise[noi_ind].transpose()[j]
                X[2 * j + 1][i][1] = self.noise[noi_ind].transpose()[j+self.data_channels]
            
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
    if True:
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
