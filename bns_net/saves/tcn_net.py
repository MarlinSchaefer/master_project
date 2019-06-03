import keras
import numpy as np
import h5py
import os
import json

#The implementation of this network is based on:
#https://arxiv.org/pdf/1803.01271.pdf
#
#There is a code to get weight normalization to work:
#https://github.com/openai/weightnorm
#
#The idea to use a TCN in the first place came from the paper provided
#by Frank on 22.05.2019 (dd.mm.yyyy) via E-Mail
#
#The idea to use the TCN for reconstruction and use the inception-net
#afterwards was original.

NUM_OF_DETECTORS = 2

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

def get_input(input_names):
    global NUM_OF_DETECTORS
    inp_sig = keras.layers.Input(shape=(4096, NUM_OF_DETECTORS), name=input_names[0])
    inp_noi = keras.layers.Input(shape=(4096, NUM_OF_DETECTORS), name=input_names[1])
    add = keras.layers.Add()([inp_sig, inp_noi])
    return((add, inp_sig, inp_noi))

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
    
    for i in range(12):
        #print("i: {}".format(i))
        
        if i == 0:
            x = tcn_residual_block(inp, k=kernel, d=2**i)
        else:
            if i == 11:
                x = tcn_residual_block(x, k=kernel, d=2**i, name=name, num_filters=2)
            else:
                x = tcn_residual_block(x, k=kernel, d=2**i)
        
        #print("x: {}".format(x))
    
    return(x)

def inception_stack(x):
    conv_1 = keras.layers.Conv1D(64, 32)(x)
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

def get_model():
    global NUM_OF_DETECTORS
    inp1, is1, in1 = get_input(("Signal_1s", "Noise_1s"))
    inp2, is2, in2 = get_input(("Signal_2s", "Noise_2s"))
    inp4, is4, in4 = get_input(("Signal_4s", "Noise_4s"))
    inp8, is8, in8 = get_input(("Signal_8s", "Noise_8s"))
    inp16, is16, in16 = get_input(("Signal_16s", "Noise_16s"))
    inp32, is32, in32 = get_input(("Signal_32s", "Noise_32s"))
    inp64, is64, in64 = get_input(("Signal_64s", "Noise_64s"))
    
    #b = keras.layers.Conv1D(1, 1)(inp1)
    
    tcn1 = get_tcn_stack(inp1, name="TCN_out_1s")
    tcn2 = get_tcn_stack(inp2, name="TCN_out_2s")
    tcn4 = get_tcn_stack(inp4, name="TCN_out_4s")
    tcn8 = get_tcn_stack(inp8, name="TCN_out_8s")
    tcn16 = get_tcn_stack(inp16, name="TCN_out_16s")
    tcn32 = get_tcn_stack(inp32, name="TCN_out_32s")
    tcn64 = get_tcn_stack(inp64, name="TCN_out_64s")
    
    add1 = keras.layers.Add()([inp1, tcn1])
    add2 = keras.layers.Add()([inp2, tcn2])
    add4 = keras.layers.Add()([inp4, tcn4])
    add8 = keras.layers.Add()([inp8, tcn8])
    add16 = keras.layers.Add()([inp16, tcn16])
    add32 = keras.layers.Add()([inp32, tcn32])
    add64 = keras.layers.Add()([inp64, tcn64])
    
    incep1 = inception_stack(add1)
    incep2 = inception_stack(add2)
    incep4 = inception_stack(add4)
    incep8 = inception_stack(add8)
    incep16 = inception_stack(add16)
    incep32 = inception_stack(add32)
    incep64 = inception_stack(add64)
    
    combined = keras.layers.concatenate([incep1, incep2, incep4, incep8, incep16, incep32, incep64])
    
    deep_incep1 = incp_lay(combined, 32)
    deep_batch1 = keras.layers.BatchNormalization()(deep_incep1)
    deep_pool1 = keras.layers.MaxPooling1D(4)(deep_batch1)
    deep_incep2 = incp_lay(deep_pool1, 32)
    deep_batch2 = keras.layers.BatchNormalization()(deep_incep2)
    dim_red = keras.layers.Conv1D(16, 1)(deep_batch2)
    
    flat = keras.layers.Flatten()(dim_red)
    
    snr_dense1 = keras.layers.Dense(2)(flat)
    snr_dense2 = keras.layers.Dense(1, name='Out_SNR')(snr_dense1)
    
    bool_dense1 = keras.layers.Dense(3)(flat)
    bool_dense2 = keras.layers.Dense(2, name='Out_bool')(bool_dense1)
    
    model = keras.models.Model(inputs=[is1, in1, is2, in2, is4, in4, is8, in8, is16, in16, is32, in32, is64, in64], outputs=[snr_dense2, bool_dense2, tcn1, tcn2, tcn4, tcn8, tcn16, tcn32, tcn64])
    
    return(model)

def compile_model(model):
    model.compile(loss=['mse', 'categorical_crossentropy', 'mse', 'mse', 'mse', 'mse', 'mse', 'mse', 'mse'], optimizer='adam', metrics={'Out_SNR': 'mape', 'Out_bool': 'accuracy'})

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
                num_signal = num_noise
            elif type(slice) in [tuple, list] and len(slice) == 2:
                num_signal = slice[0]
                num_noise = slice[1]
            else:
                raise ValueError('slice needs to be a tuple or list of exactly 2 items.')
            
            signal_indices = generate_unique_index_pairs(2*num_signals)
            noise_indices = np.arange(0, len(self.noise), dtype=int)
            np.random.shuffle(noise_indices)
            self.training_indices = np.array(signal_indices[:num_signals] + noise_indices[:num_noise])
            np.random.shuffle(self.training_indices)
            self.training_indices = [(pt[0], pt[1]) for pt in self.training_indices]
            self.testing_indices = np.array(signal_indices[num_signals:] + noise_indices[:num_noise])
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
        
        def generate_unique_index_pairs_noise(self, num_pairs):
            if len(self.noise) < num_pairs:
                raise ValueError("Can't generate more indices for pure noise than there are noise instances.")
            
            if num_pairs < len(self.noise) / 2:
                invert = False
            else:
                invert = True
                num_pairs = len(self.noise) - num_pairs
            
            curr_pairs = 0
            poss = np.zeros(len(self.noise))
            while curr_pairs < num_pairs:
                r_int = np.random.randint(0, len(self.noise))
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
        
        def generate_unique_index_pairs(self, num_pairs, generate_signals_only=True):
            len_sig = len(self.signals)
            len_noi = len(self.noise)
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
        
        y_3 = [np.zeros([len(indices), 2, self.signals[0].shape[0]]) for i in range(self.data_channels)]
        
        for i, idx in enumerate(indices):
            sig_ind, noi_ind = self.index_list[idx]
            
            for j in range(self.data_channels):
                if not sig_ind == -1:
                    X[2 * j][i][0] = self.signals[sig_ind].transpose()[j]
                    X[2 * j][i][1] = self.signals[sig_ind].transpose()[j+self.data_channels]
                    y_3[j][i] = self.signals[sig_ind].transpose()[j]
                
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
        
        y_3 = [dat.transpose(0, 2, 1) for dat in y_3]
        
        return((X, [y_1, y_2] + y_3))

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

