from pycbc.waveform import get_td_waveform
from multiprocessing import Pool
import numpy as np
from random import seed, random, uniform, randint
from functools import partial
from pycbc.psd import aLIGOZeroDetHighPower, interpolate, inverse_spectrum_truncation
from pycbc.noise import noise_from_psd
from pycbc.types.timeseries import TimeSeries
from pycbc.filter import sigma, resample_to_delta_t
import os
import h5py

"""
Splits the data into training- and testing set and stores it as .hf5 file.

Args:
    -(str)name: Complete filepath
    -(list/np.array)data: The data that should be stored
    -(int)split_index: The index at which the data is split into training- and
                       testing-set

Ret:
    -(void)
"""
def data_to_file(name, data, split_index):
    train_data = data[0][:split_index]
    train_raw = data[1][:split_index]
    train_labels = data[2][:split_index]
    
    test_data = data[0][split_index:]
    test_raw = data[1][split_index:]
    test_labels = data[2][split_index:]
    
    output = h5py.File(name, 'w')
    
    training = output.create_group('training')
    training.create_dataset('train_data', data=train_data, dtype=np.float64)
    training.create_dataset('train_labels', data=train_labels, dtype=np.float64)
    training.create_dataset('train_raw', data=train_raw, dtype=np.float64)
    
    train_parameters = training.create_group('parameters')
    train_parameters.create_dataset('wav_parameters', data=np.array(data[3][:split_index]), dtype=np.array(data[3][:split_index]).dtype)
    train_parameters.create_dataset('ext_parameters', data=np.array(data[4][:split_index]), dtype=np.array(data[4][:split_index]).dtype)
    
    
    testing = output.create_group('testing')
    testing.create_dataset('test_data', data=test_data, dtype=np.float64)
    testing.create_dataset('test_labels', data=test_labels, dtype=np.float64)
    testing.create_dataset('test_raw', data=test_raw, dtype=np.float64)
    
    test_parameters = testing.create_group('parameters')
    test_parameters.create_dataset('wav_parameters', data=np.array(data[3][split_index:]), dtype=np.array(data[3][split_index:]).dtype)
    test_parameters.create_dataset('ext_parameters', data=np.array(data[4][split_index:]), dtype=np.array(data[4][split_index:]).dtype)
    
    psd = output.create_group('psd')
    psd.create_dataset('data', data=data[5], dtype=np.float64)
    psd.create_dataset('delta_f', data=data[6], dtype=np.float64)
    
    output.close()
    return()

#TODO: Implement correctly?
def activation_function(hp, hc):
    return(hp)

"""
Generates a specified a single template. It is a helper function to
'create_file' and is called multiple times.

Args:
    -(int)i: Counter to keep track of which template is beeing generated at the
             moment.
    -kwargs: This contains all optional arguments
             pycbc.waveform.get_td_waveform can take, as well as the following,
             which are described in the documentation of 'create_file':
                +snr
                +gw_prob
                +random_delta_t
                +t_len
                +resample_t_len
                +time_variance
                +whiten_len
                +whiten_cutoff

Ret:
    -(np.array): Array containing the contents of a TimeSeries-object as
                 np.array in the first entry and the supposed SNR in the second
                 entry.
"""
def payload(i, **kwargs):
    print(i)
    opt_arg = {}
    opt_keys = ['snr', 'gw_prob', 'random_starting_time', 'resample_delta_t', 't_len', 'resample_t_len', 'time_variance', 'whiten_len', 'whiten_cutoff']
    
    for key, val in kwargs.items():
        if not key == 'mode_array':
            if type(val) == list:
                kwargs[key] = uniform(val[0], val[1])
    
    for key in opt_keys:
        try:
            opt_arg[key] = kwargs.get(key)
            del kwargs[key]
        except KeyError:
            print("The necessary argument '%s' was not supplied in 'make_template_file.py'" % key)
    
    T_SAMPLES = int(opt_arg['t_len']/kwargs['delta_t'])
    DELTA_F = 1.0/opt_arg['t_len']
    F_LEN = int(2.0/(DELTA_F * kwargs['delta_t']))
    
    gw_present = bool(random() < opt_arg['gw_prob'])
    
    psd = aLIGOZeroDetHighPower(length=F_LEN, delta_f=DELTA_F, low_freq_cutoff=kwargs['f_lower'])
    noise = noise_from_psd(length=T_SAMPLES, delta_t=kwargs['delta_t'], psd=psd, seed=randint(0,100000))
    
    if gw_present:
        #print(kwargs)
        hp, hc = get_td_waveform(**kwargs)
        
        strain = activation_function(hp, hc)
        
        if opt_arg['random_starting_time']:
            t_before = int(round((T_SAMPLES) / 2)) - len(strain.sample_times) + int(round(random() * opt_arg['time_variance'] * strain.sample_rate))
        else:
            t_before = int(round((T_SAMPLES) / 2)) - len(strain.sample_times)
            #t_before = int(round((T_SAMPLES - len(strain)) / 2))
        
        strain.prepend_zeros(t_before)
        strain.append_zeros(T_SAMPLES - len(strain))
        #print("SNR: %f" % opt_arg['snr'])
        #print("Scaling factor: %f" % (sigma(strain, psd=psd, low_frequency_cutoff=kwargs['f_lower'])))
        kwargs['distance'] = sigma(strain, psd=psd, low_frequency_cutoff=kwargs['f_lower']) / opt_arg['snr']
        strain /= sigma(strain, psd=psd, low_frequency_cutoff=kwargs['f_lower'])
        strain *= opt_arg['snr']
    else:
        strain = TimeSeries(np.zeros(len(noise)))
        opt_arg['snr'] = 0.0
    
    noise._epoch = strain._epoch
    full = TimeSeries(noise + strain)
    total = TimeSeries(noise + strain)
    del noise
    del strain
        
    total = total.whiten(opt_arg['whiten_len'], opt_arg['whiten_cutoff'], low_frequency_cutoff=kwargs['f_lower'])
        
    mid_point = (total.end_time + total.start_time) / 2
    total = resample_to_delta_t(total, opt_arg['resample_delta_t'])
    full = resample_to_delta_t(full, opt_arg['resample_delta_t'])
    total_crop = total.time_slice(mid_point-opt_arg['resample_t_len']/2, mid_point+opt_arg['resample_t_len']/2)
    full_crop  = full.time_slice(mid_point-opt_arg['resample_t_len']/2, mid_point+opt_arg['resample_t_len']/2)
    
    del full
    del total
    
    return((np.array(total_crop), np.array(full_crop), opt_arg['snr'], str(kwargs), str(opt_arg)))

"""
Create a template file using the given options.

Creates a template file with the name specified in name. (extension '.hf5')
It takes all keyword-arguments pycbc.waveform.get_td_waveform can take and some
more, which are specified below. Each argument for
pycbc.waveform.get_td_waveform can also be given as a list. If an argument is
given as a list, the function will uniformly distribute the value between the
minimum and the maximum of this list. (Valid for all parameters, but useful
only for source parameters, such as mass or spin.)
The signal to noise ratio is scaled by dividing the generated signal by its
sigma value (for details on this see the documentation of pycbc.filter.sigma).2
Therefore changing the distance from the default will result in inaccurate
SNRs.
This function also sets some default source parameters, if no others are
provided. They are the following:
    -approximant: "SEOBNRv4_opt"
    -mass1: 30.0
    -mass2: 30.0
    -delta_t: 1.0 / 4096
    -f_lower: 20.0
    -coa_phase: [0., np.pi]
    -distance: 1.0

Args:
    -(str)name: Filename of the finished template file (without extension).
    -(op,float/list)snr: The SNR(-range) to generate the templates at.
                         Default: [1.0,12.0]
    -(op,float)gw_prob: How probable the function is to generate a signal
                        within the noise background. (*) Default: 1.0
    -(op,bool)random_starting_time: Wether or not the position of a potential
                                    signal will be varied within the
                                    timeseries. (**) Default: True
    -(op,float)time_variance: Time (in s) the position of the GW within the2
                              signal will maximally be varied by. Default: 1.0
    -(op,float)resample_delta_t: The sample frequency the templates will be
                                 stored at. Default: 1.0 / 1024
    -(op,float)t_len: Duration of the total signal + noise segment in seconds.
                      (*3) Default: 64.0
    -(op,float)resample_t_len: The duration in seconds of the signal that is
                               stored. Default: 4.0
    -(op,float)whiten_len: See documentation of
                           pycbc.timeseries.TimeSeries.whiten for details.
                           Default: 4.0
    -(op,float)whiten_cutoff: See documentation of
                              pycbc.timeseries.TimeSeries.whiten for details.
                              Default: 4.0
    -(op,int)num_of_templates: How many templates the final file will contain
                               Default: 20000
    -(op,int)seed: The seed that is used by the PRNG provided by the random
                   module. Default: 12345
    -(op,float)train_to_test: Ratio of the number of training samples over the
                              total number of templates. This values MUST be
                              between 0.0 and 1.0. Default: 0.7
    -(op,str)path: The absolute (or relative) path of where the template file
                   will be stored at. Default: ""
    -(op,tuple)data_shape: The shape each entry in the data should have.
    -(op,tuple)label_shape: The shape each entry for the labels should have

Ret:
    -(void)

Notes:
    -(*): This variable should be in the range of [0,1]. If a value larger than
          1 is given, it will act as if gw_prob was set to 1, if a value
          smaller then 0 is given it will act as if gw_prob was set to 0.
          In case no GW is present within the signal, the SNR will be set to 0.
    -(**): In case this option set to True, the signal will only be pushed
           forward. The maximum amount it will be pushed by is specified in
           'time_variance'.
    -(*3): This is not the length of the template that is being stored in the
           end! The algorithm generates a lot more signal then it outputs,
           because it has to filter and trim the data during the process of
           creating usable data. Therefore it is advised to chose a time t_len
           which is a lot longer then the final duration of the signal that is
           stored.
    -This function runs in parallel utilizing multiprocessing.Pool. Therefore
     the user MUST make sure it is invoked with the condition
    
        if __name__ == "__main__":
            create_file(name, **kwargs)
"""
def create_file(name, **kwargs):
    wav_arg = {}
    opt_arg = {}
    
    #Properties the payload function needs
    #Properties for the waveform itself
    wav_arg['approximant'] = "SEOBNRv4_opt"
    wav_arg['mass1'] = 30.0
    wav_arg['mass2'] = 30.0
    wav_arg['delta_t'] = 1.0 / 4096
    wav_arg['f_lower'] = 20.0
    wav_arg['coa_phase'] = [0., np.pi]
    wav_arg['distance'] = 1.0
    
    #Properties for handeling the process of generating the waveform
    wav_arg['snr'] = [1.0, 12.0]
    wav_arg['gw_prob'] = 1.0
    wav_arg['random_starting_time'] = True
    wav_arg['time_variance'] = 1.0
    wav_arg['resample_delta_t'] = 1.0 / 1024
    wav_arg['t_len'] = 64.0
    wav_arg['resample_t_len'] = 4.0
    wav_arg['whiten_len'] = 4.0
    wav_arg['whiten_cutoff'] = 4.0
    
    for key in wav_arg.keys():
        if key in kwargs:
            wav_arg[key] = kwargs.get(key)
            del kwargs[key]
    
    #Properties for the generating program
    opt_arg['num_of_templates'] = 20000
    opt_arg['seed'] = 12345
    opt_arg['train_to_test'] = 0.7
    opt_arg['path'] = ""
    opt_arg['data_shape'] = (int(1.0 / wav_arg['delta_t']),)
    opt_arg['label_shape'] = (1,)
    
    for key in opt_arg.keys():
        if key in kwargs:
            opt_arg[key] = kwargs.get(key)
            del kwargs[key]
    
    for key, val in wav_arg.items():
        if not key == 'mode_array':
            if type(val) == list:
                wav_arg[key] = [min(val), max(val)]
    
    kwargs.update(wav_arg)
    seed(opt_arg['seed'])
    
    pool = Pool()
    data = pool.map(partial(payload, **kwargs), range(opt_arg['num_of_templates']))
    
    T_SAMPLES = int(kwargs['t_len']/kwargs['delta_t'])
    DELTA_F = 1.0/kwargs['t_len']
    F_LEN = int(2.0/(DELTA_F * kwargs['delta_t']))
    
    data_0 = np.array([x[0] for x in data])
    data_1 = np.array([x[1]for x in data])
    data_2 = np.array([x[2] for x in data])
    data_3 = [x[3] for x in data]
    data_4 = [x[4] for x in data]
    data_5 = np.array(aLIGOZeroDetHighPower(length=F_LEN, delta_f=DELTA_F, low_freq_cutoff=kwargs['f_lower']))
    data_6 = DELTA_F
    
    del data
    
    in_shape = list(opt_arg['data_shape'])
    in_shape.insert(0, opt_arg['num_of_templates'])
    
    out_shape = list(opt_arg['label_shape'])
    out_shape.insert(0, opt_arg['num_of_templates'])
    
    data_0 = data_0.reshape(in_shape)
    data_1 = data_1.reshape(in_shape)
    data_2 = data_2.reshape(out_shape)
    
    data = [data_0, data_1, data_2, data_3, data_4, data_5, data_6]
    
    data_to_file(name=os.path.join(opt_arg['path'], name + '.hf5'), data=data, split_index=int(round(opt_arg['train_to_test']*opt_arg['num_of_templates'])))
