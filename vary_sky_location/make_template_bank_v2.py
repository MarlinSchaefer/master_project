from pycbc.waveform import get_td_waveform
import multiprocessing as mp
import numpy as np
from random import seed, random, uniform, randint
from functools import partial
from pycbc.psd import aLIGOZeroDetHighPower, interpolate, inverse_spectrum_truncation
from pycbc.noise import noise_from_psd
from pycbc.types.timeseries import TimeSeries
from pycbc.filter import sigma, resample_to_delta_t
import os
import h5py
from run_net import filter_keys

"""
TODO: Implement this function which should return a list of dictionaries
      containing the source parameters appropriatly named
"""
def generate_psd(**kwargs):
    DELTA_F = 1.0/kwargs['t_len']
    F_LEN = int(2.0/(DELTA_F * kwargs['delta_t']))
    return(aLIGOZeroDetHighPower(length=F_LEN, delta_f=DELTA_F, low_freq_cutoff=kwargs['f_lower']))

def generate_parameters(num_of_templates, rand_seed, **kwargs):
    seed(rand_seed)
    
    #print(kwargs)
    
    ret = []
    
    tmp_dic = {}
    
    for i in range(num_of_templates):
        #print(kwargs)
        #print(ret)
        for key, val in kwargs.items():
            #print("{}, {}".format(key,val))
            if not key == 'mode_array':
                if type(val) == list:
                    #print("If  :{}: {}".format(key, uniform(val[0], val[1])))
                    tmp_dic[key] = uniform(val[0], val[1])
                    #print("If dic: {}: {}".format(key, tmp_dic[key]))
                else:
                    #print("Else: {}: {}".format(key, val))
                    tmp_dic[key] = val
                    #print("Else dic: {}: {}".format(key, tmp_dic[key]))
            else:
                tmp_dic[key] = val
        
        ret.append(dict(tmp_dic))
    
    #print(ret)
    #for dic in ret:
        #print(dic['snr'])
    #print('Exiting generation')
    return(ret)

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
def worker(kwargs):
    #print('Entering worker')
    #print(kwargs)
    full_kwargs = dict(kwargs)
    kwargs = dict(kwargs)
    
    opt_arg = {}
    opt_keys = ['snr', 'gw_prob', 'random_starting_time', 'resample_delta_t', 't_len', 'resample_t_len', 'time_variance', 'whiten_len', 'whiten_cutoff']
    
    
    for key in opt_keys:
        try:
            opt_arg[key] = kwargs.get(key)
            del kwargs[key]
        except KeyError:
            print("The necessary argument '%s' was not supplied in '%s'" % (key, str(__file__)))
            #exit()
    
    T_SAMPLES = int(opt_arg['t_len']/kwargs['delta_t'])
    DELTA_F = 1.0/opt_arg['t_len']
    F_LEN = int(2.0/(DELTA_F * kwargs['delta_t']))
    
    gw_present = bool(random() < opt_arg['gw_prob'])
    
    psd = generate_psd(**full_kwargs)
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
        
        opt_arg['samples_before'] = t_before
        
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
        opt_arg['samples_before'] = 0
    
    noise._epoch = strain._epoch
    full = TimeSeries(noise + strain)
    total = TimeSeries(noise + strain)
    del noise
    del strain
        
    total = total.whiten(opt_arg['whiten_len'], opt_arg['whiten_cutoff'], low_frequency_cutoff=kwargs['f_lower'])
        
    mid_point = (total.end_time + total.start_time) / 2
    total = resample_to_delta_t(total, opt_arg['resample_delta_t'])
    #full = resample_to_delta_t(full, opt_arg['resample_delta_t'])
    total_crop = total.time_slice(mid_point-opt_arg['resample_t_len']/2, mid_point+opt_arg['resample_t_len']/2)
    full_crop  = full.time_slice(mid_point-opt_arg['resample_t_len']/2, mid_point+opt_arg['resample_t_len']/2)
    
    
    #del full
    #del total
    
    #print(opt_arg)
    
    return((np.vstack(np.array(total_crop)), np.array(full), np.array([opt_arg['snr']]), np.array(str(kwargs)), np.array(str(opt_arg))))

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
    
    wav_arg, kwargs = filter_keys(wav_arg, kwargs)
    
    for key, val in wav_arg.items():
        if not key == 'mode_array':
            if type(val) == list:
                wav_arg[key] = [min(val), max(val)]
    
    #Properties for the generating program
    opt_arg['num_of_templates'] = 20000
    opt_arg['seed'] = 12345
    opt_arg['train_to_test'] = 0.7
    opt_arg['path'] = ""
    opt_arg['data_shape'] = (int(1.0 / wav_arg['delta_t']),)
    opt_arg['label_shape'] = (1,)
    
    opt_arg, kwargs = filter_keys(opt_arg, kwargs)
    
    num_of_templates = opt_arg['num_of_templates']
    
    kwargs.update(wav_arg)
    
    seed(opt_arg['seed'])
    
    parameter_list = generate_parameters(num_of_templates, opt_arg['seed'], **kwargs)
    
    
    file_name = os.path.join(opt_arg['path'], name + '.hf5')
    pool = mp.Pool()
    
    prop_dict = {}
    prop_dict.update(wav_arg)
    prop_dict.update(opt_arg)
    
    split_index = int(round(opt_arg['train_to_test'] * num_of_templates))
    
    tmp_sample = worker(parameter_list[0])
    #train_snr = np.array([pt['snr'] for pt in parameter_list[:split_index]])
    #test_snr = np.array([pt['snr'] for pt in parameter_list[split_index:]])
    
    with h5py.File(file_name, 'w') as output:
        training = output.create_group('training')
        testing = output.create_group('testing')
        psd = output.create_group('psd')
        parameter_space = output.create_group('parameter_space')
        train_parameters = training.create_group('parameters')
        test_parameters = testing.create_group('parameters')
        
        gen_psd = generate_psd(**kwargs)
        #Is np.float64 enough?
        psd.create_dataset('data', data=np.array(gen_psd), dtype=np.float64)
        psd.create_dataset('delta_f', data=gen_psd.delta_f, dtype=np.float64)
        
        for key, val in prop_dict.items():
            parameter_space.create_dataset(str(key), data=np.array(val), dtype=np.array(val).dtype)
        
        print((tmp_sample[0]).shape)
        print((split_index, (tmp_sample[0]).shape[0], (tmp_sample[0]).shape[1]))
        #Assumes the data to be in shape (time_samples, 1)
        train_data = training.create_dataset('train_data', shape=(split_index, (tmp_sample[0]).shape[0], (tmp_sample[0]).shape[1]), dtype=tmp_sample[0].dtype)
        
        #Assumes the data to be in shape (time_samples, )
        train_raw = training.create_dataset('train_raw', shape=(split_index, (tmp_sample[1]).shape[0], ), dtype=tmp_sample[1].dtype)
        
        #Needs the SNR to be a single number. This has to be returned as the
        #second entry and as a numpy array of shape '()'
        train_labels = training.create_dataset('train_labels', shape=(split_index, 1), dtype=tmp_sample[2].dtype)
        
        #Assumes the shape () for the provided data
        train_wav_parameters = train_parameters.create_dataset('wav_parameters', shape=(split_index, ), dtype=tmp_sample[3].dtype)
        train_ext_parameters = train_parameters.create_dataset('ext_parameters', shape=(split_index, ), dtype=tmp_sample[4].dtype)
        
        
        #Assumes the data to be in shape (time_samples, 1)
        test_data = testing.create_dataset('test_data', shape=(num_of_templates - split_index, (tmp_sample[0]).shape[0], (tmp_sample[0]).shape[1]), dtype=tmp_sample[0].dtype)
        
        #Assumes the data to be in shape (time_samples, )
        test_raw = testing.create_dataset('test_raw', shape=(num_of_templates - split_index, (tmp_sample[1]).shape[0], ), dtype=tmp_sample[1].dtype)
        
        #Needs the SNR to be a single number. This has to be returned as the
        #second entry and as a numpy array of shape '()'
        test_labels = testing.create_dataset('test_labels', shape=(num_of_templates - split_index, 1), dtype=tmp_sample[2].dtype)
        
        #Assumes the shape () for the provided data
        test_wav_parameters = test_parameters.create_dataset('wav_parameters', shape=(num_of_templates - split_index, ), dtype=tmp_sample[3].dtype)
        test_ext_parameters = test_parameters.create_dataset('ext_parameters', shape=(num_of_templates - split_index, ), dtype=tmp_sample[4].dtype)
        
        
        for idx, dat in enumerate(pool.imap_unordered(worker, parameter_list)):
            print(idx)
            if idx < split_index:
                #write to training
                i = idx
                train_data[i] = dat[0]
                train_raw[i] = dat[1]
                train_labels[i] = dat[2]
                train_wav_parameters[i] = dat[3]
                train_ext_parameters[i] = dat[4]
            else:
                #write to testing
                i = idx - num_of_templates
                test_data[i] = dat[0]
                test_raw[i] = dat[1]
                test_labels[i] = dat[2]
                test_wav_parameters[i] = dat[3]
                test_ext_parameters[i] = dat[4]
        
        
    
    
    
    
    """
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
    #data_1 = data_1.reshape(in_shape)
    print(data_1.shape)
    data_2 = data_2.reshape(out_shape)
    
    prop_dict = {}
    prop_dict.update(wav_arg)
    prop_dict.update(opt_arg)
    for key in prop_dict.keys():
        prop_dict[key] = np.array(prop_dict[key])
    
    data = [data_0, data_1, data_2, data_3, data_4, data_5, data_6, prop_dict]
    
    data_to_file(name=os.path.join(opt_arg['path'], name + '.hf5'), data=data, split_index=int(round(opt_arg['train_to_test']*opt_arg['num_of_templates'])))
    """