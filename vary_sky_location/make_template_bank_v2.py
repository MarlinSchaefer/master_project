from pycbc.waveform import get_td_waveform
import multiprocessing as mp
import numpy as np
from random import seed, random, uniform, randint
from functools import partial
from pycbc.psd import aLIGOZeroDetHighPower, interpolate, inverse_spectrum_truncation
from pycbc.noise import noise_from_psd
from pycbc.types.timeseries import TimeSeries
from pycbc.filter import sigma, resample_to_delta_t, sigmasq, matched_filter
import os
import h5py
from run_net import filter_keys
from pycbc.detector import Detector
import sys
from progress_bar import progress_tracker

"""
TODO: Implement this function which should return a list of dictionaries
      containing the source parameters appropriatly named
"""
def generate_psd(**kwargs):
    DELTA_F = 1.0/kwargs['t_len']
    F_LEN = int(2.0/(DELTA_F * kwargs['delta_t']))
    return(aLIGOZeroDetHighPower(length=F_LEN, delta_f=DELTA_F, low_freq_cutoff=kwargs['f_lower']))

def generate_parameters(num_of_templates, rand_seed, **kwargs):
    exceptions = ['mode_array', 'detectors']
    seed(rand_seed)
    
    #print(kwargs)
    
    ret = []
    
    tmp_dic = {}
    
    for i in range(num_of_templates):
        #print(kwargs)
        #print(ret)
        for key, val in kwargs.items():
            #print("{}, {}".format(key,val))
            if not key in exceptions:
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

def detector_projection(hp, hc, **kwargs):
    """Returns the waveform projected onto different detectors.
    
    Arguments
    ---------
    hp : TimeSeries
        TimeSeries object containing the "plus" polarization of a GW
    hc : TimeSeries
        TimeSeries object containing the "cross" polarization of a GW
    
    Returns
    -------
    list
        A list containing the signals projected onto the detectors specified in
        in kwargs['detectors].
    """
    end_time = kwargs['end_time']
    detectors = kwargs['detectors']
    declination = kwargs['declination']
    right_ascension = kwargs['right_ascension']
    polarization = kwargs['polarization']
    
    del kwargs['end_time']
    del kwargs['detectors']
    del kwargs['declination']
    del kwargs['right_ascension']
    del kwargs['polarization']
    
    detectors = [Detector(d) for d in detectors]
    
    hp.start_time += end_time
    hc.start_time += end_time
    
    ret = [d.project_wave(TimeSeries(hp), TimeSeries(hc), right_ascension, declination, polarization) for d in detectors]
    
    return(ret)

def set_temp_offset(sig_list, t_len, t_offset):
    #Sanity check
    if not isinstance(sig_list, list) or not isinstance(sig_list[0], type(TimeSeries([0], 0.1))):
        raise TypeError("A list of pycbc.types.timeseries.TimeSeries objects must be provided.")
        sys.exit(0)
    
    dt = sig_list[0].delta_t
    
    for pt in sig_list:
        if not pt.delta_t == dt:
            raise ValueError("The timeseries must have the same delta_t!")
            sys.exit(0)
    
    #Take the first signal as reference and apply the timeshift in respect to
    #this signal (so only this template will be centered at t=0 for a timeshift
    #of 0)
    ref = sig_list[0].sample_times[0]
    
    #Calculate the temporal offset between the templates
    prep_list = [ref - dat.sample_times[0] for dat in sig_list]
    
    #Find the signal that happens the earliest and store its offset to the
    #reference signal (the earliest signal will have a negative offset)
    min_val = min(prep_list)
    
    #Calculate the offset of every template with respect to this earliest
    prep_list = [dat - min_val for dat in prep_list]
    
    #Convert time to samples
    prep_list = [int(dat / dt) for dat in prep_list]
    
    #Calculate for every signal how many zeros have to be prepended for the
    #first signal to have a temporal offset of T_OFFSET and the other signals
    #to stay in relation to this first signal
    prep_list = [int(t_len / (2 * dt)) - len(sig_list[0]) - dat + int(t_offset / dt) for dat in prep_list]
    
    for i, dat in enumerate(sig_list):
        #Prepend the zeros calculated before
        dat.prepend_zeros(prep_list[i])
        #Append as many zeros as needed to get to a final length of T_LEN seconds
        dat.append_zeros(int(t_len / dt) - len(dat))
    
    return

def rescale_to_snr(sig_list, snr, psd, f_lower):
    """Rescale the list of signals by the total detector SNR.
    
    Parameters
    ----------
    sig_list : list
        List of the strain data (as TimeSeries object) for every detector
    snr : float
        The desiered SNR over all detectors
    psd : FrequencySeries
        The PSD that is used in all detectors
    f_lower : float
        Low frequency cutoff of the signals and the PSD
    
    Returns
    -------
    list
        A list of the rescaled strain data in the order it was given in sig_list
    """
    snrsq_list = [sigmasq(strain, psd=psd, low_frequency_cutoff=f_lower) for strain in sig_list]
    div = np.sqrt(sum(snrsq_list))
    return([pt / div * snr for pt in sig_list])

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
    #print("Worker here!")
    #print("Args: {}".format(kwargs))
    full_kwargs = dict(kwargs)
    kwargs = dict(kwargs)
    
    opt_arg = {}
    opt_keys = ['snr', 'gw_prob', 'random_starting_time', 'resample_delta_t', 't_len', 'resample_t_len', 'time_offset', 'whiten_len', 'whiten_cutoff']
    
    for key in opt_keys:
        try:
            opt_arg[key] = kwargs.get(key)
            del kwargs[key]
        except KeyError:
            print("The necessary argument '%s' was not supplied in '%s'" % (key, str(__file__)))
    
    projection_arg = {}
    projection_arg['end_time'] = 1337 * 137 * 42
    projection_arg['declination'] = 0.0
    projection_arg['right_ascension'] = 0.0
    projection_arg['polarization'] = 0.0
    projection_arg['detectors'] = ['L1', 'H1']
    
    projection_arg, kwargs = filter_keys(projection_arg, kwargs)
    
    T_SAMPLES = int(opt_arg['t_len']/kwargs['delta_t'])
    DELTA_F = 1.0/opt_arg['t_len']
    F_LEN = int(2.0/(DELTA_F * kwargs['delta_t']))
    
    gw_present = bool(random() < opt_arg['gw_prob'])
    
    psd = generate_psd(**full_kwargs)
    #TODO: Generate the seed for this prior to parallelizing
    noise_list = [noise_from_psd(length=T_SAMPLES, delta_t=kwargs['delta_t'], psd=psd, seed=randint(0,100000)) for d in projection_arg['detectors']]
    
    #print("Pre GW generation")
    
    if gw_present:
        #Generate waveform
        #print("Pre waveform")
        hp, hc = get_td_waveform(**kwargs)
        
        #Project it onto the considered detectors (This could be handeled using)
        #a list, to make room for more detectors
        #print("Pre projection")
        strain_list = detector_projection(TimeSeries(hp), TimeSeries(hc), **projection_arg)
        
        #Enlarge the signals bya adding zeros infront and after. Take care of a
        #random timeshift while still keeping the relative timeshift between
        #detectors
        #TODO: This should also be set outside
        if opt_arg['random_starting_time']:
            t_offset = opt_arg['time_offset']
        else:
            t_offset = 0.0
        
        #print("Pre embedding in zero")
        set_temp_offset(strain_list, opt_arg['t_len'], t_offset)
        
        #Rescale the templates to match wanted SNR
        strain_list = rescale_to_snr(strain_list, opt_arg['snr'], psd, kwargs['f_lower'])
    else:
        strain_list = [TimeSeries(np.zeros(len(noise_list[0]))) for n in range(len(noise_list))]
        opt_arg['snr'] = 0.0
    
    #print("post generating")
    total_white = []
    matched_snr_sq = []
    #NOTE: JUST HERE FOR MAKING THIS WORK
    total_l = []
    #print("Pre loop")
    for i, noise in enumerate(noise_list):
        #print("Loop i: {}".format(i))
        #Add strain to noise
        noise._epoch = strain_list[i]._epoch
        #print("Post epoch, pre adding")
        total = TimeSeries(noise + strain_list[i])
        #NOTE: JUST HERE FOR MAKING THIS WORK
        total_l.append(total)
        #print("Post adding, pre whiten")
        
        #Whiten the total data, downsample and crop the data
        total_white.append(total.whiten(opt_arg['whiten_len'], opt_arg['whiten_cutoff'], low_frequency_cutoff=kwargs['f_lower']))
        #print("Post whiten and appending, pre resampling")
        total_white[i] = resample_to_delta_t(total_white[i], opt_arg['resample_delta_t'])
        #print("Post resampling, pre cropping")
        mid_point = (total_white[i].end_time + total_white[i].start_time) / 2
        total_white[i] = total_white[i].time_slice(mid_point-opt_arg['resample_t_len']/2, mid_point+opt_arg['resample_t_len']/2)
        #print("Post cropping, pre matched filtering")
        #print("Strain list: {}\ntotal: {}\nPSD: {}".format(strain_list[i], total, psd))
        
        #test = matched_filter(strain_list[i], total, psd=psd, low_frequency_cutoff=kwargs['f_lower'])
        #print("Can calc")
        #NOTE: TAKEN OUT TO MAKE THIS WORK
        #Calculate matched filter snr
        #matched_snr_sq.append(max(abs(matched_filter(strain_list[i], total, psd=psd, low_frequency_cutoff=kwargs['f_lower'])))**2)
        #print("Post matched filtering, WTF!")
    
    #del total
    #del strain_list
    
    #Calculate the total SNR of all detectors
    #NOTE: TAKEN OUT TO MAKE THIS WORK
    #calc_snr = np.sqrt(sum(matched_snr_sq))
    #del matched_snr_sq
    
    out_wav = [[dat[i] for dat in total_white] for i in range(len(total_white[0]))]
    
    #print("Pre return")
    
    #NOTE: TAKEN OUT TO MAKE THIS WORK
    #return((np.array(out_wav), np.array([opt_arg['snr']]), np.array(calc_snr), np.array(str(kwargs)), np.array(str(opt_arg))))
    #NOTE: JUST HERE FOR MAKING THIS WORK
    return((np.array(out_wav), np.array([opt_arg['snr']]), strain_list, np.array(str(kwargs)), np.array(str(opt_arg)), total_l))

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
    #print("Hello world!")
    wav_arg = {}
    opt_arg = {}
    
    #Properties the payload function needs
    #Properties for the waveform itself
    wav_arg['approximant'] = "SEOBNRv4_opt"
    wav_arg['mass1'] = 30.0
    wav_arg['mass2'] = 30.0
    wav_arg['delta_t'] = 1.0 / 4096
    wav_arg['f_lower'] = 20.0
    wav_arg['coa_phase'] = [0., 2 * np.pi]
    wav_arg['distance'] = 1.0
    
    #Properties for handeling the process of generating the waveform
    wav_arg['snr'] = [6.0, 15.0]
    wav_arg['gw_prob'] = 1.0
    wav_arg['random_starting_time'] = True
    wav_arg['time_offset'] = [-0.5, 0.5]
    wav_arg['resample_delta_t'] = 1.0 / 1024
    wav_arg['t_len'] = 64.0
    wav_arg['resample_t_len'] = 4.0
    wav_arg['whiten_len'] = 4.0
    wav_arg['whiten_cutoff'] = 4.0
    
    #Skyposition
    wav_arg['end_time'] = 1337 * 137 * 42
    wav_arg['declination'] = 0.0
    wav_arg['right_ascension'] = 0.0
    wav_arg['polarization'] = 0.0
    wav_arg['detectors'] = ['L1', 'H1']
    
    """
    #These are just here to remember the values each one of these can take
    wav_arg['declination'] = [-np.pi / 2, np.pi / 2]
    wav_arg['right_ascension'] = [-np.pi, np.pi]
    wav_arg['polarization'] = [0.0, np.pi]
    """
    
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
    
    #tmp_sample = worker(parameter_list[0])
    
    pool = mp.Pool()
    
    prop_dict = {}
    prop_dict.update(wav_arg)
    prop_dict.update(opt_arg)
    
    split_index = int(round(opt_arg['train_to_test'] * num_of_templates))
    #train_snr = np.array([pt['snr'] for pt in parameter_list[:split_index]])
    #test_snr = np.array([pt['snr'] for pt in parameter_list[split_index:]])
    
    tmp_sample = worker(parameter_list[0])
    
    #print("Pre file")
    
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
        
        #Assumes the data to be in shape ()
        #NOTE: TAKEN OUT TO MAKE THIS WORK
        #train_snr_calculated = training.create_dataset('train_snr_calculated', shape=(split_index, ), dtype=tmp_sample[2].dtype)
        #NOTE: JUST HERE TO MAKE THIS WORK
        train_snr_calculated = training.create_dataset('train_snr_calculated', shape=(split_index, ), dtype=np.float64)
        
        #Needs the SNR to be a single number. This has to be returned as the
        #second entry and as a numpy array of shape '()'
        train_labels = training.create_dataset('train_labels', shape=(split_index, 1), dtype=tmp_sample[1].dtype)
        
        #Assumes the shape () for the provided data
        train_wav_parameters = train_parameters.create_dataset('wav_parameters', shape=(split_index, ), dtype=tmp_sample[3].dtype)
        train_ext_parameters = train_parameters.create_dataset('ext_parameters', shape=(split_index, ), dtype=tmp_sample[4].dtype)
        
        
        #Assumes the data to be in shape (time_samples, 1)
        test_data = testing.create_dataset('test_data', shape=(num_of_templates - split_index, (tmp_sample[0]).shape[0], (tmp_sample[0]).shape[1]), dtype=tmp_sample[0].dtype)
        
        #Assumes the data to be in shape ()
        #NOTE: TAKEN OUT TO MAKE THIS WORK
        #test_snr_calculated = testing.create_dataset('test_snr_calculated', shape=(num_of_templates - split_index, ), dtype=tmp_sample[2].dtype)
        #NOTE: JUST HERE TO MAKE THIS WORK
        test_snr_calculated = testing.create_dataset('test_snr_calculated', shape=(num_of_templates - split_index, ), dtype=np.float64)
        
        #Needs the SNR to be a single number. This has to be returned as the
        #second entry and as a numpy array of shape '()'
        test_labels = testing.create_dataset('test_labels', shape=(num_of_templates - split_index, 1), dtype=tmp_sample[1].dtype)
        
        #Assumes the shape () for the provided data
        test_wav_parameters = test_parameters.create_dataset('wav_parameters', shape=(num_of_templates - split_index, ), dtype=tmp_sample[3].dtype)
        test_ext_parameters = test_parameters.create_dataset('ext_parameters', shape=(num_of_templates - split_index, ), dtype=tmp_sample[4].dtype)
        
        #print("Pre pool")
    
        #pool = mp.Pool()
        
        #print("Number of workers: {}".format(pool._processes))
        
        #print("Pre loop")
        #print("Parameter List: {}".format(parameter_list))
        
        bar = progress_tracker(num_of_templates, name='Generating templates')
        
        for idx, dat in enumerate(pool.imap_unordered(worker, parameter_list)):
        #for idx, dat in enumerate(map(worker, parameter_list)):
            if idx < split_index:
                #NOTE: JUST HERE TO MAKE THIS WORK
                strain_list = dat[2]
                total_l = dat[5]
                matched_snr_sq = [max(abs(matched_filter(strain_list[i], total_l[i], psd=gen_psd, low_frequency_cutoff=kwargs['f_lower'])))**2 for i in range(len(strain_list))]
                calc_snr = np.sqrt(sum(matched_snr_sq))
                strain_list = None
                total_l = None
                
                #NOTE: WAS HERE PREVEIOUSLY
                #write to training
                i = idx
                train_data[i] = dat[0]
                train_labels[i] = dat[1]
                #NOTE: TAKEN OUT TO MAKE THIS WORK
                #train_snr_calculated[i] = dat[2]
                #NOTE: JUST HERE TO MAKE THIS WORK
                train_snr_calculated[i] = calc_snr
                train_wav_parameters[i] = dat[3]
                train_ext_parameters[i] = dat[4]
            else:
                #NOTE: JUST HERE TO MAKE THIS WORK
                strain_list = dat[2]
                total_l = dat[5]
                matched_snr_sq = [max(abs(matched_filter(strain_list[i], total_l[i], psd=gen_psd, low_frequency_cutoff=kwargs['f_lower'])))**2 for i in range(len(strain_list))]
                calc_snr = np.sqrt(sum(matched_snr_sq))
                strain_list = None
                total_l = None
                
                #write to testing
                i = idx - num_of_templates
                test_data[i] = dat[0]
                test_labels[i] = dat[1]
                #NOTE: TAKEN OUT TO MAKE THIS WORK
                #test_snr_calculated[i] = dat[2]
                #NOTE: JUST HERE TO MAKE THIS WORK
                test_snr_calculated[i] = calc_snr
                test_wav_parameters[i] = dat[3]
                test_ext_parameters[i] = dat[4]
            
            bar.iterate()
        
        #print("Closing parallel")
        
        pool.close()
        pool.join()
        return
