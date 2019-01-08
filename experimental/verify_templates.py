import h5py
from pycbc.waveform import get_td_waveform
from pycbc.filter import matched_filter
from pycbc.types.frequencyseries import FrequencySeries
from pycbc.types.timeseries import TimeSeries
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.psd import aLIGOZeroDetHighPower
import ast
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import normaltest

def load_templates(file_path):
    inp = h5py.File(file_path, "r")
    
    data_psd = inp['psd']['data'].value
    delta_f_psd = inp['psd']['delta_f'].value
    psd = FrequencySeries(data_psd, delta_f=delta_f_psd)
    
    train_data = inp['training']['train_data'].value
    train_labels = inp['training']['train_labels'].value
    train_raw = inp['training']['train_raw'].value
    
    tr_wav_par = inp['training']['parameters']['wav_parameters']
    tr_ext_par = inp['training']['parameters']['ext_parameters']
    
    train_wav_par = [ast.literal_eval(dic) for dic in tr_wav_par]
    
    train_ext_par = [ast.literal_eval(dic) for dic in tr_ext_par]
    
    
    
    test_data = inp['testing']['test_data'].value
    test_labels = inp['testing']['test_labels'].value
    test_raw = inp['testing']['test_raw'].value
    
    te_wav_par = inp['testing']['parameters']['wav_parameters']
    te_ext_par = inp['testing']['parameters']['ext_parameters']
    
    test_wav_par = [ast.literal_eval(dic) for dic in te_wav_par]
    
    test_ext_par = [ast.literal_eval(dic) for dic in te_ext_par]

    
    inp.close()
    
    return(((train_data, train_raw, train_labels, train_wav_par, train_ext_par),(test_data, test_raw, test_labels, test_wav_par, test_ext_par), psd))

def whiten(ts, psd, f_low):
    return((ts.to_frequencyseries() / psd**0.5).to_timeseries())

def verify_templates(file_path):
    
    #Can I merge train_data and test_data (and all the according parameters)?
    (train_data, train_raw, train_labels, train_wav_par, train_ext_par), (test_data, test_raw, test_labels, test_wav_par, test_ext_par), psd = load_templates(file_path)
    
    print(type(psd.delta_f))
    
    snr_res = []
    
    for i in range(len(train_data)):
        
        true_wav = get_td_waveform(**(train_wav_par[i]))[0]
        true_wav.resize(len(train_data[i]))
        
        psd = interpolate(psd, true_wav.delta_f)
        psd.resize(len(true_wav)/2+1)
        psd = inverse_spectrum_truncation(psd, len(psd), low_frequency_cutoff=train_wav_par[i]['f_lower'])
        
        #print(true_wav)
        #true_wav = whiten(true_wav, psd, train_wav_par[i]['f_lower'])
        #print(true_wav)
        
        snr_res.append((max(abs(matched_filter(true_wav, TimeSeries(train_data[i].transpose()[0], true_wav.delta_t), psd=psd, low_frequency_cutoff=train_wav_par[i]['f_lower']))), max(abs(matched_filter(true_wav, TimeSeries(train_raw[i].transpose()[0], true_wav.delta_t), psd=psd, low_frequency_cutoff=train_wav_par[i]['f_lower']))), train_labels[i][0]))#Is this correct. Can I take max(abs(TimeSeries))? Is that well defined?
    
    #print(train_data[0])
    
    diff = [pt[1] - pt[2] for pt in snr_res]
    mean = sum(diff) / len(diff)
    for i in range(len(diff)):
        diff[i] -= mean
    
    bin_borders = np.linspace(-10,10,21)
    bins = np.zeros(20)
    
    for pt in diff:
        for j in range(len(bins)):
            if pt < bin_borders[j+1] and pt >= bin_borders[j]:
                bins[j] += 1
    
    
    
    #binned = np.digitize(bins,np.arange(-9-5,10.5,1.0))
    
    plt.show(plt.bar(np.arange(-9.5,10.5,1.0),bins))
    
    stats, pval = normaltest(diff)
    
    return((snr_res, pval))

