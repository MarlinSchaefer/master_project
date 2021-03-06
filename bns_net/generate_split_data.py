#from tensorflow import ConfigProto, Session
#config = ConfigProto()
#config.gpu_options.allow_growth = True
#session = Session(config=config)
#import keras
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
from pycbc.psd import inverse_spectrum_truncation, interpolate
#from evaluate_nets import evaluate_training
import json
import time

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
    #dic['t_from_right'] = 2.25
    dic['t_from_right'] = 4.25
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
    org_type = type(strain_list)
    if not org_type == list:
        strain_list = [strain_list]
    
    #set to strain[0].delta_f
    DF = 1.0 / strain_list[0].delta_t / (4 * strain_list[0].sample_rate) #This is the definition of delta_f from the TimeSeries.whiten in the welchs method.
    #set to len(strain_list[0]) / 2 + 1
    F_LEN = int(4 * strain_list[0].sample_rate / 2 + 1)
    get_hyper_waveform_defaults
    low_freq_diff = 20
    if low_freq_cutoff > low_freq_diff:
        tmp_psd = aLIGOZeroDetHighPower(length=F_LEN, delta_f=DF, low_freq_cutoff=low_freq_cutoff-low_freq_diff)
    else:
        tmp_psd = aLIGOZeroDetHighPower(length=F_LEN, delta_f=DF, low_freq_cutoff=0)

    for i in range(len(strain_list)):
        max_filter_len = int(4 * strain_list[i].sample_rate) #To replicate TimeSeries.whiten
        tmp_psd_2 = interpolate(tmp_psd, strain_list[i].delta_f) #To replicate TimeSeries.whiten
        tmp_psd_2 = inverse_spectrum_truncation(tmp_psd_2, max_filter_len=max_filter_len, low_frequency_cutoff=low_freq_cutoff, trunc_method='hann')
        strain_list[i] = (strain_list[i].to_frequencyseries() / tmp_psd_2 ** 0.5).to_timeseries()
        strain_list[i] = strain_list[i][int(float(max_filter_len)/2):int(len(strain_list[i])-float(max_filter_len)/2)] #To replicate TimeSeries.whiten

    if not org_type == list:
        return(strain_list[0])
    else:
        return(strain_list)

def whiten_data_new(strain_list, low_freq_cutoff=20., max_filter_duration=4., psd=None):
    org_type = type(strain_list)
    if not org_type == list:
        strain_list = [strain_list]
    
    ret = []
    for strain in strain_list:
        df = strain.delta_f
        f_len = int(len(strain) / 2) + 1
        if psd is None:
            psd = aLIGOZeroDetHighPower(length=f_len, delta_f=df, low_freq_cutoff=low_freq_cutoff-2.)
        else:
            if not len(psd) == f_len:
                msg = 'Length of PSD does not match data.'
                raise ValueError(msg)
            elif not psd.delta_f == df:
                psd = interpolate(psd, df)
        max_filter_len = int(max_filter_duration * strain.sample_rate) #Cut out the beginning and end
        psd = inverse_spectrum_truncation(psd, max_filter_len=max_filter_len, low_frequency_cutoff=low_freq_cutoff, trunc_method='hann')
        f_strain = strain.to_frequencyseries()
        kmin = int(low_freq_cutoff / df)
        f_strain.data[:kmin] = 0
        f_strain.data[-1] = 0
        f_strain.data[kmin:] /= psd[kmin:] ** 0.5
        strain = f_strain.to_timeseries()
        ret.append(strain[max_filter_len:len(strain)-max_filter_len])
    
    if not org_type == list:
        return(ret[0])
    else:
        return(ret)

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
    #print(projection_parameters)
    strain_list = detector_projection(hp, hc, **projection_parameters)

    #Set temporal offset
    set_temp_offset(strain_list, hyper_parameters['t_len'], hyper_parameters['time_offset'], hyper_parameters['t_from_right'])

    #Rescale to total SNR
    strain_list = rescale_to_snr(strain_list, hyper_parameters['snr'], psd, waveform_parameters['f_lower'])

    #Whiten the signal here
    strain_list = whiten_data_new(strain_list, psd=None, low_freq_cutoff=waveform_parameters['f_lower'])

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
    strain_list = [noise_from_psd(length=T_LEN, delta_t=dt, psd=psd, seed=seed[i]) for i in range(num_detectors)]

    #Whiten noise
    strain_list = whiten_data_new(strain_list, psd=None, low_freq_cutoff=f_lower)
    
    #ret = resample_data(strain_list, sample_rates)

    return(resample_data(strain_list, sample_rates))

def generate_template(file_path, num_pure_signals, num_pure_noise, sample_rates=[4096, 2048, 1024, 512, 256, 128, 64], reduced=False, **kwargs):
    #Manually setting some defaults
    if not 'seed' in kwargs:
        kwargs['seed'] = 0

    if not 't_len' in kwargs:
        kwargs['t_len'] = 96.0

    if not 'f_lower' in kwargs:
        kwargs['f_lower'] = 20.0

    if not 'detectors' in kwargs:
        kwargs['detectors'] = ['L1', 'H1']

    if not 'no_gw_snr' in kwargs:
        kwargs['no_gw_snr'] = 4.0
    
    parameters = generate_parameters(num_pure_signals, rand_seed=kwargs['seed'], **kwargs)
    print(parameters)
    
    for dic in parameters:
        dic['sample_rates'] = sample_rates
    
    noise_seeds = np.random.randint(0, 10**8, num_pure_noise * len(kwargs['detectors']))
    noise_seeds = noise_seeds.reshape((num_pure_noise, len(kwargs['detectors'])))
    
    pool = mp.Pool()

    with h5py.File(file_path, 'w') as FILE:
        if reduced:
            data_shape = (num_pure_signals, 2048, 2 * (len(sample_rates) + 1))
        else:
            data_shape = (num_pure_signals, 4096, 2 * len(sample_rates))
        signals = FILE.create_group('signals')
        signal_data = signals.create_dataset('data', shape=data_shape, dtype=np.float64)
        signal_snr = signals.create_dataset('snr', shape=(num_pure_signals, ), dtype=np.float64)
        signal_bool = signals.create_dataset('bool', shape=(num_pure_signals, ), dtype=np.float64)
        if reduced:
            data_shape = (num_pure_noise, 2048, 2 * (len(sample_rates) + 1))
        else:
            data_shape = (num_pure_noise, 4096, 2 * len(sample_rates))
        noise = FILE.create_group('noise')
        noise_data = noise.create_dataset('data', shape=data_shape, dtype=np.float64)
        noise_snr = noise.create_dataset('snr', shape=(num_pure_noise, ), dtype=np.float64)
        noise_bool = noise.create_dataset('bool', shape=(num_pure_noise, ), dtype=np.float64)
        parameter_space = FILE.create_group('parameter_space')
        if not 'snr' in kwargs:
            kwargs['snr'] = [8.0, 15.0]
        for k, v in kwargs.items():
            parameter_space.create_dataset(str(k), data=np.array(v), dtype=np.array(v).dtype)

        bar = progress_tracker(num_pure_signals, name='Generating signals')
        
        if reduced:
            #Something
            X = np.zeros(data_shape[1:]).transpose()
            for i, dat in enumerate(pool.imap_unordered(signal_worker, parameters)):
            #for i, dat in enumerate(list(map(signal_worker, parameters))):
                tmp_dat = dat[0].transpose()
                for j in range(len(sample_rates) + 1):
                    if j == 0:
                        X[j] = tmp_dat[j][2048:]
                        X[j+len(sample_rates)+1] = tmp_dat[j+len(sample_rates)][2048:]
                    else:
                        X[j] = tmp_dat[j-1][:2048]
                        X[j+len(sample_rates)+1] = tmp_dat[(j-1)+len(sample_rates)][:2048]
                signal_data[i] = X.transpose()
                signal_snr[i] = dat[1]
                signal_bool[i] = 1.0

                bar.iterate()
        else:
            for i, dat in enumerate(pool.imap_unordered(signal_worker, parameters)):
            #for i, dat in enumerate(list(map(signal_worker, parameters))):
                signal_data[i] = dat[0]
                signal_snr[i] = dat[1]
                signal_bool[i] = 1.0

                bar.iterate()

        bar = progress_tracker(num_pure_noise, name='Generating noise')
        
        if reduced:
            #Something
            X = np.zeros(data_shape[1:]).transpose()
            for i, dat in enumerate(pool.imap_unordered(noise_worker, [(kwargs['t_len'], kwargs['f_lower'], 1.0 / max(sample_rates), len(kwargs['detectors']), noise_seeds[i], sample_rates) for i in range(num_pure_noise)])):
            #for i, dat in enumerate(list(map(noise_worker, [(kwargs['t_len'], kwargs['f_lower'], 1.0 / max(sample_rates), len(kwargs['detectors']), np.random.randint(0, 10**8), sample_rates) for i in range(num_pure_noise)]))):
                tmp_dat = dat[0].transpose()
                for j in range(len(sample_rates) + 1):
                    if j == 0:
                        X[j] = tmp_dat[j][2048:]
                        X[j+len(sample_rates)+1] = tmp_dat[j+len(sample_rates)][2048:]
                    else:
                        X[j] = tmp_dat[j-1][:2048]
                        X[j+len(sample_rates)+1] = tmp_dat[(j-1)+len(sample_rates)][:2048]
                noise_data[i] = X.transpose()
                noise_snr[i] = kwargs['no_gw_snr']
                noise_bool[i] = 0.0

                bar.iterate()
        else:
            for i, dat in enumerate(pool.imap_unordered(noise_worker, [(kwargs['t_len'], kwargs['f_lower'], 1.0 / max(sample_rates), len(kwargs['detectors']), noise_seeds[i], sample_rates) for i in range(num_pure_noise)])):
            #for i, dat in enumerate(list(map(noise_worker, [(kwargs['t_len'], kwargs['f_lower'], 1.0 / max(sample_rates), len(kwargs['detectors']), np.random.randint(0, 10**8), sample_rates) for i in range(num_pure_noise)]))):
                noise_data[i] = dat
                noise_snr[i] = kwargs['no_gw_snr']
                noise_bool[i] = 0.0

                bar.iterate()
        
        #for i, dat in enumerate(pool.imap_unordered(noise_worker, [(kwargs['t_len'], kwargs['f_lower'], 1.0 / max(sample_rates), len(kwargs['detectors']), noise_seeds[i], sample_rates) for i in range(num_pure_noise)])):
        ##for i, dat in enumerate(list(map(noise_worker, [(kwargs['t_len'], kwargs['f_lower'], 1.0 / max(sample_rates), len(kwargs['detectors']), np.random.randint(0, 10**8), sample_rates) for i in range(num_pure_noise)]))):
            #noise_data[i] = dat
            #noise_snr[i] = kwargs['no_gw_snr']
            #noise_bool[i] = 0.0

            #bar.iterate()

    pool.close()
    pool.join()
    return(file_path)
