import keras
import numpy as np
import multiprocessing as mp
import h5py
import os
from make_template_bank_bns import detector_projection, set_temp_offset, generate_parameters, rescale_to_snr
from progress_bar import progress_tracker
from aux_functions import filter_keys, get_store_path
from pycbc.waveform import get_td_waveform
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.filter import resample_to_delta_t
from pycbc.noise import noise_from_psd
from data_object import DataSet
from generator import DataGenerator
from pycbc.psd import inverse_spectrum_truncation
from evaluate_nets import evaluate_training
import json

def get_waveform_default():
    dic = {}
    dic['approximant'] = 'TaylorF2'
    dic['mass1'] = 1.4
    dic['mass2'] = 1.4
    dic['delta_t'] = 1.0 / 4096
    dic['f_lower'] = 20.0
    dic['coa_phase'] = 0.0
    dic['distance'] = 1.0
    return(dic)

def get_hyper_waveform_defaults():
    dic = {}
    dic['snr'] = 10.0
    dic['t_from_right'] = 0.5
    dic['time_offset'] = 0.0
    dic['t_len'] = 96.0
    return(dic)

def get_projection_defaults():
    projection_arg = {}
    projection_arg['end_time'] = 1337 * 137 * 42
    projection_arg['declination'] = 0.0
    projection_arg['right_ascension'] = 0.0
    projection_arg['polarization'] = 0.0
    projection_arg['detectors'] = ['L1', 'H1']
    return(projection_arg)

def resample_data(strain_list, sample_rates):
    if not type(strain_list) == list:
        strain_list = [strain_list]
    
    ret = np.zeros((len(strain_list) * len(sample_rates), 4096))
    
    for i, dat in enumerate(strain_list):
        for j, sr in enumerate(sample_rates):
            idx = i * len(sample_rates) + j
            ret[idx] = np.array(resample_to_delta_t(dat, 1.0 / sr).data[-4096:])
    
    ret = ret.transpose()
    
    return(ret)

def whiten_data(strain_list, psd, low_freq_cutoff=20.0):
    """
    print('strain_list[0]:\n\tdt: {}\n\tdf: {}'.format(strain_list[0].delta_t, strain_list[0].delta_f))
    print('psd:\n\tdt: {}\n\tdf: {}'.format(psd.delta_t, psd.delta_f))
    org_type = type(strain_list)
    if not org_type == list:
        strain_list = [strain_list]
    
    tmp_psd = interpolate(psd, strain_list[0].delta_f)
    print('tmp_psd:\n\tdt: {}\n\tdf: {}'.format(tmp_psd.delta_t, tmp_psd.delta_f))
    
    for i in range(len(strain_list)):
        strain_list[i] = (strain_list[i].to_frequencyseries() / tmp_psd ** 0.5).to_timeseries()
    
    print('Whitened')
    
    if not org_type == list:
        return(strain_list[0])
    else:
        return(strain_list)
    """
    org_type = type(strain_list)
    if not org_type == list:
        strain_list = [strain_list]
    
    DF = strain_list[0].delta_f
    F_LEN = len(strain_list[0].to_frequencyseries())
    tmp_psd = inverse_spectrum_truncation(aLIGOZeroDetHighPower(length=F_LEN, delta_f=DF, low_freq_cutoff=low_freq_cutoff), max_filter_len=len(strain_list[0]), low_frequency_cutoff=low_freq_cutoff, trunc_method='hann')
    
    for i in range(len(strain_list)):
        strain_list[i] = (strain_list[i].to_frequencyseries() / tmp_psd ** 0.5).to_timeseries()
    
    if not org_type == list:
        return(strain_list[0])
    else:
        return(strain_list)

def signal_worker(parameters):
    waveform_parameters, parameters = filter_keys(get_waveform_default(), parameters)
    hyper_parameters, parameters = filter_keys(get_hyper_waveform_defaults(), parameters)
    projection_parameters, parameters = filter_keys(get_projection_defaults(), parameters)
    sample_rates = parameters['sample_rates']
    
    #Generate PSD
    DF = 1.0 / hyper_parameters['t_len']
    F_LEN = int(2.0 / (DF * waveform_parameters['delta_t']))
    psd = aLIGOZeroDetHighPower(length=F_LEN, delta_f=DF, low_freq_cutoff=waveform_parameters['f_lower'])
    
    #Generate waveform
    hp, hc = get_td_waveform(**waveform_parameters)
    
    #Project waveform onto projectors
    strain_list = detector_projection(hp, hc, **projection_parameters)
    
    #Set temporal offset
    set_temp_offset(strain_list, hyper_parameters['t_len'], hyper_parameters['time_offset'], hyper_parameters['t_from_right'])
    
    #Rescale to total SNR
    strain_list = rescale_to_snr(strain_list, hyper_parameters['snr'], psd, waveform_parameters['f_lower'])
    
    #Whiten the signal here
    strain_list = whiten_data(strain_list, psd, low_freq_cutoff=waveform_parameters['f_lower'])
    
    #Resample appropriately
    return((resample_data(strain_list, sample_rates), hyper_parameters['snr']))

def noise_worker(parameters):
    #Name parameters
    t_len = parameters[0]
    f_lower = parameters[1]
    dt = parameters[2]
    num_detectors = parameters[3]
    seed = parameters[4]
    sample_rates = parameters[5]
    
    DF = 1.0 / t_len
    F_LEN = int(2.0 / (DF * dt))
    
    #Generate PSD
    psd = aLIGOZeroDetHighPower(length=F_LEN, delta_f=DF, low_freq_cutoff=f_lower)
    
    T_LEN = int(t_len / dt)
    
    #Generate noise
    strain_list = [noise_from_psd(length=T_LEN, delta_t=dt, psd=psd, seed=seed) for i in range(num_detectors)]
    
    #Whiten noise
    strain_list = whiten_data(strain_list, psd, low_freq_cutoff=f_lower)
    
    return(resample_data(strain_list, sample_rates))

def generate_template(file_path, num_pure_signals, num_pure_noise, sample_rates=[4096, 2048, 1024, 512, 256, 128, 64], **kwargs):
    #Manually setting some defaults
    if not 'seed' in kwargs:
        kwargs['seed'] = 0
    parameters = generate_parameters(num_pure_signals, rand_seed=kwargs['seed'], **kwargs)
    
    if not 't_len' in kwargs:
        kwargs['t_len'] = 96.0
    
    if not 'f_lower' in kwargs:
        kwargs['f_lower'] = 20.0
    
    if not 'detectors' in kwargs:
        kwargs['detectors'] = ['L1', 'H1']
    
    if not 'no_gw_snr' in kwargs:
        kwargs['no_gw_snr'] = 4.0
    
    for dic in parameters:
        dic['sample_rates'] = sample_rates
    
    pool = mp.Pool()
    
    with h5py.File(file_path, 'w') as FILE:
        signals = FILE.create_group('signals')
        signal_data = signals.create_dataset('data', shape=(num_pure_signals, 4096, 2 * len(sample_rates)), dtype=np.float64)
        signal_snr = signals.create_dataset('snr', shape=(num_pure_signals, ), dtype=np.float64)
        signal_bool = signals.create_dataset('bool', shape=(num_pure_signals, ), dtype=np.float64)
        noise = FILE.create_group('noise')
        noise_data = noise.create_dataset('data', shape=(num_pure_noise, 4096, 2 * len(sample_rates)), dtype=np.float64)
        noise_snr = noise.create_dataset('snr', shape=(num_pure_noise, ), dtype=np.float64)
        noise_bool = noise.create_dataset('bool', shape=(num_pure_noise, ), dtype=np.float64)
        
        bar = progress_tracker(num_pure_signals, name='Generating signals')
        
        for i, dat in enumerate(pool.imap_unordered(signal_worker, parameters)):
        #for i, dat in enumerate(list(map(signal_worker, parameters))):
            signal_data[i] = dat[0]
            signal_snr[i] = dat[1]
            signal_bool[i] = 1.0
            
            bar.iterate()
        
        bar = progress_tracker(num_pure_noise, name='Generating noise')
        
        for i, dat in enumerate(pool.imap_unordered(noise_worker, [(kwargs['t_len'], kwargs['f_lower'], 1.0 / max(sample_rates), len(kwargs['detectors']), np.random.randint(0, 10**8), sample_rates) for i in range(num_pure_noise)])):
        #for i, dat in enumerate(list(map(noise_worker, [(kwargs['t_len'], kwargs['f_lower'], 1.0 / max(sample_rates), len(kwargs['detectors']), np.random.randint(0, 10**8), sample_rates) for i in range(num_pure_noise)]))):
            noise_data[i] = dat
            noise_snr[i] = kwargs['no_gw_snr']
            noise_bool[i] = 0.0
            
            bar.iterate()
    
    pool.close()
    pool.join()
    return(file_path)

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

def get_model(num_detectors):
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

def get_generator():
    class DataGenerator(keras.utils.Sequence):
        #Here we need to hack a bit. The function 'store_test_results' expects
        #to get a dobj with templates and labels. So we will load all stuff into
        #the template part of the custom dobj defined below and feed it this.
        #Together with this modified generator it should swallow the data.
        def __init__(self, dobj_loaded_test_data, dobj_loaded_test_labels, batch_size=32, shuffle=True):
            
            (signals, noise, signal_labels, signal_in_noise_list, noise_snr) = dobj_loaded_test_data
            
            self.signals = signals
            self.signal_labels = signal_labels
            self.noise = noise
            self.indices = []
            self.signal_in_noise_list = signal_in_noise_list
            self.batch_size = batch_size
            self.shuffle = True
            self.combinations = []
            self.snrs = []
            self.noise_snr = noise_snr
            
            self.on_epoch_end()
        
        def __len__(self):
            return(int(np.ceil(float(len(self.signal_in_noise_list)) / self.batch_size)))
        
        def __getitem__(self, index):
            if (index+1) * self.batch_size > len(self.signal_in_noise_list):
                fetch = self.signal_in_noise_list[index*self.batch_size:]
            else:
                fetch = self.signal_in_noise_list[index*self.batch_size:(index+1)*self.batch_size]
            
            min_ind = self.curr_signal_index
            max_ind = self.curr_signal_index+fetch.count(True)
            if not max_ind < len(self.signal_indices):
                signal_indices = list(self.signal_indices[min_ind:])
                self.reset_signal_index()
                max_ind = max_ind - len(self.signal_indices) + 1
                signal_indices += list(self.signal_indices[0:max_ind])
                signal_indices = np.array(signal_indices)
                self.curr_signal_index += max_ind
            else:
                signal_indices = self.signal_indices[min_ind:max_ind]
                self.curr_signal_index += fetch.count(True)
            
            min_ind = self.curr_noise_index
            max_ind = self.curr_noise_index+len(fetch)
            if not max_ind < len(self.noise_indices):
                noise_indices = list(self.noise_indices[min_ind:])
                self.reset_noise_index()
                max_ind = max_ind - len(self.noise_indices) + 1
                noise_indices += list(self.noise_indices[0:max_ind])
                noise_indices = np.array(noise_indices)
                self.curr_noise_index += max_ind
            else:
                noise_indices = self.noise_indices[min_ind:max_ind]
                self.curr_noise_index += len(fetch)
            
            X, y = self.__data_generation(fetch, signal_indices, noise_indices)
            
            return(X, y)
        
        def __data_generation(self, fetch, signal_indices, noise_indices):
            num_of_channels = self.signals[0].shape[-1]
            
            X = [np.empty([len(fetch), 2] + [self.signals[0].shape[0]])] * num_of_channels
            
            y_1 = np.empty((len(fetch), 1))
            
            y_2 = np.empty((len(fetch), 2))
            
            sig_counter = 0
            
            for i, do_fetch in enumerate(fetch):
                
                noi_idx = noise_indices[i]
                tmp_noise = self.noise[noi_idx].transpose()
                
                if do_fetch:
                    sig_idx = signal_indices[sig_counter]
                    tmp_sig = self.signals[sig_idx].transpose()
                    y_1[i] = self.signal_labels[i]
                    y_2[i] = np.array([1., 0.])
                    self.combinations.append((sig_idx, noi_idx))
                    sig_counter += 1
                else:
                    tmp_sig = np.zeros(self.signals[0].transpose().shape)
                    y_1[i] = self.noise_snr
                    y_2[i] = np.array([0., 1.])
                    self.combinations.append((-1, noi_idx))
                
                self.snrs.append(y_1[i])
                for k in range(num_of_channels): #Which input channel to use
                    if k % 2 == 0:#Process signal
                        X[k][i][0] = tmp_sig[k/2]
                        X[k][i][1] = tmp_sig[k/2+num_of_channels/2]
                    else:
                        X[k][i][0] = tmp_noise[int(np.floor(float(k)/2))]
                        X[k][i][1] = tmp_sig[int(np.floor(float(k)/2))+num_of_channels/2]
            
            X = [dat.transpose(0, 2, 1) for dat in X]
            
            return(X, [y_1, y_2])
                        
        
        def reset_signal_index(self):
            self.signal_indices = np.arange(len(self.signals))
            if self.shuffle:
                np.random.shuffle(self.signal_indices)
            self.curr_signal_index = 0
        
        def reset_noise_index(self):
            self.noise_indices = np.arange(len(self.noise))
            if self.shuffle:
                np.random.shuffle(self.noise_indices)
            self.curr_noise_index = 0
        
        def on_epoch_end(self):
            self.reset_signal_index()
            self.reset_noise_index()
            self.last_combinations = self.combinations
            self.combinations = []
        
        def get_indices(self):
            return((self.last_combinations, self.combinations))
        
        def get_snr_list(self):
            return(self.snrs)
        
        def get_bool_list(self):
            return(self.signal_in_noise_list)
    ###
    
    return(DataGenerator)

def get_dobj(file_path, signal_in_noise_list):
    #This is a real dirty hack to the data_object.DataSet. It has only the
    #necessary methods as required by evaluate_nets. So this will NOT work in
    #the general construct of run_net.
    class CustomDataSet():
        def __init__(self, file_path, signal_in_noise_list):
            self.file_path = file_path
            self.signal_in_noise_list = signal_in_noise_list
            self.snrs = []
            self.load_data()
        
        def load_data(self):
            with h5py.File(self.file_path) as FILE:
                self.signals = FILE['signals']['data'][:]
                self.noise = FILE['noise']['data'][:]
                self.signal_labels = FILE['signals']['snr'][:]
                self.noise_label = FILE['noise']['snr'][0]
        
        @property
        def loaded_test_data(self):
            return((self.signals, self.noise, self.signal_labels, self.signal_in_noise_list, self.noise_label))
        
        @property
        def loaded_test_labels(self):
            if self.snrs == []:
                return([])
            else:
                ret = [np.array(self.snrs), np.array([[1.0, 0.0] if self.signal_in_noise_list[i] else [0.0, 1.0] for i in range(len(self.snrs))])]
        
        def set_snrs(self, labels):
            self.snrs = labels
            self.loaded_test_snr = np.zeros(len(self.snrs))
    
    return(CustomDataSet(file_path, signal_in_noise_list))

#generate_template(file_path, num_pure_signals, num_pure_noise, sample_rates=[4096, 2048, 1024, 512, 256, 128, 64], **kwargs):

def main():
    #Create data
    dir_name = 'addititve_net_results'
    template_path = os.path.join(get_store_path(), dir_name, 'templates.hf5')
    #generate_template(template_path, 200, 1000, snr=[8.0, 15.0])
    
    #Load data
    num_total_samples = 100000
    signal_in_noise_list = [np.random.random() < 0.5 for i in range(num_total_samples)]
    #signal_in_noise_list = [False for i in range(num_total_samples)]
    #signal_in_noise_list = [True, False]
    dobj = get_dobj(template_path, signal_in_noise_list)
    
    generator = get_generator()
    generator = generator(dobj.loaded_test_data, dobj.loaded_test_labels, batch_size=32)
    print("Max of Signal indices: {}".format(max(generator.signal_indices)))
    
    #Load network (correctly! Need the weights from a previous try and initiate it with that.)
    net = get_model(2)
    
    net.compile(loss={'Out_SNR': 'mean_squared_error', 'Out_Bool': 'categorical_crossentropy'}, loss_weights={'Out_SNR': 1.0, 'Out_Bool': 0.5}, optimizer='adam', metrics={'Out_SNR': 'mape', 'Out_Bool': 'accuracy'})
    
    #print(net.predict_generator(generator))
    
    print(net.summary())
    num_of_epochs = 1
    epochs_per_test = 1
    results = []
    labels = []
    for i in np.arange(0, 200, epochs_per_test):
        net.fit_generator(generator, epochs=epochs_per_test, max_queue_size=10)
        net.save(os.path.join(get_store_path(), dir_name, 'additive_net_epoch_' + str(i) + '.hf5'))
	results.append([i+epochs_per_test, net.evaluate_generator(generator)])
        labels.append([i+epochs_per_test, generator.get_indices()[0]])
    
    net.save(os.path.join(get_store_path(), dir_name, 'additive_net.hf5'))
    
    with open(os.path.join(get_store_path(), dir_name, 'results.json')) as jsonFILE:
        json.dump(results, jsonFILE, indent=4)
    with open(os.path.join(get_store_path(), dir_name, 'indices.json')) as indexFILE:
        json.dump(labels, indexFILE, indent=4)

    dobj.set_snrs(generator.get_snr_list())
    
    #net_name, dobj, dir_path, t_start, batch_size=32, generator=g.DataGeneratorMultInput, **kwargs
    evaluate_training(additive_net, dobj, os.path.join(get_store_path(), dir_name), generator=generator, show_snr_plot=False, show_false_alarm=False, show_sensitivity_plot=False, make_loss_plot=False)
    
    return

main()
