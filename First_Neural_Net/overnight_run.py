from pycbc.noise import noise_from_psd
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.waveform import get_td_waveform
from pycbc.filter import sigma, resample_to_delta_t
import numpy as np
from random import random, seed, randint
from pycbc.types.timeseries import TimeSeries
from multiprocessing import Pool
import h5py
import time
import matplotlib.pyplot as plt

#Global constants definitions
NUM_OF_TEMPLATE = 10000
SCALING_FACTOR = 6.1
RESAMPLE_DELTA_T = 1.0 / 1024
FILE_NAME_1 = "small_snr_templates"
FILE_NAME_2 = "varied_time_templates"
RANDOM_START_TIME = False
SNR_SMALL = True

APPROXIMANT = "SEOBNRv4_opt"
DELTA_T = 1.0 / 4096
F_LOWER = 20.0
TIME_LENGTH = 64.0
DELTA_F = 1.0 / TIME_LENGTH
MASS1 = 30.0
MASS2 = 30.0
F_LEN = int(2 / (DELTA_T * DELTA_F))
T_SAMPLES = int(TIME_LENGTH / DELTA_T)
GW_PROB = 1.0

_hp, _hc = get_td_waveform(approximant=APPROXIMANT, mass1=MASS1, mass2=MASS2, delta_t=DELTA_T, f_lower=F_LOWER, distance=1)
NULL_WAVEFORM = _hp

SAMPLE_POINTS = len(NULL_WAVEFORM.sample_times)

seed(12345)

def save_to_file_1(data):
    split_index = int(round(0.7 * len(data[0])))
    
    train_data = np.vstack(data[0][:split_index])
    train_labels = np.vstack(data[1][:split_index])
    
    test_data = np.vstack(data[0][split_index:])
    test_labels = np.vstack(data[1][split_index:])
    
    output = h5py.File(FILE_NAME_1 + '.hdf5', 'w')
    
    training = output.create_group('training')
    training.create_dataset('train_data', data=train_data, dtype='f')
    training.create_dataset('train_labels', data=train_labels, dtype='f')
    
    testing = output.create_group('testing')
    testing.create_dataset('test_data', data=test_data)
    testing.create_dataset('test_labels', data=test_labels)
    
    output.close()
    return()

def save_to_file_2(data):
    split_index = int(round(0.7 * len(data[0])))
    
    train_data = np.vstack(data[0][:split_index])
    train_labels = np.vstack(data[1][:split_index])
    
    test_data = np.vstack(data[0][split_index:])
    test_labels = np.vstack(data[1][split_index:])
    
    output = h5py.File(FILE_NAME_2 + '.hdf5', 'w')
    
    training = output.create_group('training')
    training.create_dataset('train_data', data=train_data, dtype='f')
    training.create_dataset('train_labels', data=train_labels, dtype='f')
    
    testing = output.create_group('testing')
    testing.create_dataset('test_data', data=test_data)
    testing.create_dataset('test_labels', data=test_labels)
    
    output.close()
    return()

def payload_1(i):
    gw_present = bool(random() < GW_PROB)
    
    snr_goal = random() * 11 + 1
    
    psd = aLIGOZeroDetHighPower(length=F_LEN, delta_f=DELTA_F, low_freq_cutoff=F_LOWER)
    noise = noise_from_psd(length=T_SAMPLES, delta_t=DELTA_T, psd=psd, seed=randint(0,10000))
    
    if gw_present:
        strain = TimeSeries(NULL_WAVEFORM)
        
        t_start = int(round((T_SAMPLES - SAMPLE_POINTS) / 2))
        
        print("i: %d, snr: %.2f, t_start: %d" % (i, snr_goal, t_start))
        
        strain.prepend_zeros(t_start)
        strain.append_zeros(T_SAMPLES-len(strain))
        
        scal = sigma(strain, psd=psd, low_frequency_cutoff=F_LOWER)
        scal /= snr_goal
        
        strain /= scal
    else:
        snr_goal = -1.0
        strain = TimeSeries([0], delta_t=DELTA_T)
        strain.append_zeros(T_SAMPLES - 1)
    
    noise._epoch = strain._epoch
    total = noise + strain
    del noise
    del strain
    total = total.whiten(4,4,low_frequency_cutoff=F_LOWER)
    
    mid_point = (total.end_time + total.start_time) / 2
    total = resample_to_delta_t(total, RESAMPLE_DELTA_T)
    total_crop = total.time_slice(mid_point-2, mid_point+2)
    
    return(np.array([np.array(total_crop), snr_goal]))

def payload_2(i):
    gw_present = bool(random() < GW_PROB)
    
    snr_goal = random() * 5 + 7
    
    psd = aLIGOZeroDetHighPower(length=F_LEN, delta_f=DELTA_F, low_freq_cutoff=F_LOWER)
    noise = noise_from_psd(length=T_SAMPLES, delta_t=DELTA_T, psd=psd, seed=randint(0,10000))
    
    if gw_present:
        strain = TimeSeries(NULL_WAVEFORM)
        
        t_start = int(round((T_SAMPLES - SAMPLE_POINTS) / 2)) + int(round(random() * strain.sample_rate))
        
        print("i: %d, snr: %.2f, t_start: %d" % (i, snr_goal, t_start))
        
        strain.prepend_zeros(t_start)
        strain.append_zeros(T_SAMPLES-len(strain))
        
        scal = sigma(strain, psd=psd, low_frequency_cutoff=F_LOWER)
        scal /= snr_goal
        
        strain /= scal
    else:
        snr_goal = -1.0
        strain = TimeSeries([0], delta_t=DELTA_T)
        strain.append_zeros(T_SAMPLES - 1)
    
    noise._epoch = strain._epoch
    total = noise + strain
    del noise
    del strain
    total = total.whiten(4,4,low_frequency_cutoff=F_LOWER)
    
    mid_point = (total.end_time + total.start_time) / 2
    total = resample_to_delta_t(total, RESAMPLE_DELTA_T)
    total_crop = total.time_slice(mid_point-2, mid_point+2)
    
    return(np.array([np.array(total_crop), snr_goal]))

def main():
    pool = Pool()
    """
    data = pool.map(payload_1, range(NUM_OF_TEMPLATE))
    
    data = np.array(data)
    data = data.transpose()
    
    save_to_file_1(data)
    
    print("Set 1 finished")
    """
    
    data = pool.map(payload_2, range(2 * NUM_OF_TEMPLATE))
    
    data = np.array(data)
    data = data.transpose()
    
    save_to_file_2(data)
    
    """
    plt.subplot(511)
    plt.plot(data[0][0])
    plt.subplot(512)
    plt.plot(data[0][1])
    plt.subplot(513)
    plt.plot(data[0][2])
    plt.subplot(514)
    plt.plot(data[0][3])
    plt.subplot(515)
    plt.plot(data[0][4])
    plt.show()
    """

if(__name__ == "__main__"):
    main()
