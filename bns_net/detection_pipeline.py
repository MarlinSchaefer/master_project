from pycbc.noise import noise_from_string
from pycbc.filter import resample_to_delta_t, matched_filter
from pycbc.waveform import get_td_waveform
from pycbc.psd import interpolate
from make_template_bank_bns import detector_projection, set_temp_offset, rescale_to_snr
import random as r
from run_net import get_store_path
import matplotlib.pyplot as plt
import os
import numpy as np
import keras
from pycbc.types.timeseries import TimeSeries
import time
from progress_bar import progress_tracker
from datetime import datetime, timedelta
import multiprocessing as mp
import ctypes as c
from contextlib import closing

def load_data():
    path = '~/Downloads/tseries.hdf'
    with h5py.File(path, 'r') as FILE:
        raw = [FILE['L1'].value, FILE['H1'].value]
    
    return([TimeSeries(d, delta_t=1.0/4096) for d in raw])

def find_max(ts):
    curr_max = [-np.inf, -np.inf]
    for i in range(len(ts)):
        if ts[i] > curr_max[1]:
            curr_max[0] = ts.sample_times[i]
            curr_max[1] = ts[i]
    return(curr_max)

def resample(ts):
    time_slices = [TimeSeries(ts.data[len(ts)-int(i * ts.sample_rate):], delta_t=ts.delta_t) for i in [1, 2, 4, 8, 16, 32, 64]]
    res_slices = []
    for i, t in enumerate(time_slices):
        res_slices.append(list(resample_to_delta_t(t, 1.0 / 2 ** (12 - i))))
        #The following line was added to make it suite my current setup
        res_slices.append(list(np.zeros(len(res_slices[-1]))))
    del time_slices
    return(res_slices)

def init(mp_arr_, aux_info_):
    global mp_arr
    global aux_info
    
    mp_arr = tonumpyarray(mp_arr_)
    aux_info = tonumpyarray(aux_info_)

def tonumpyarray(arr):
    return np.frombuffer(arr.get_obj())

def get_slice(offset):
    #print("Offset: {}".format(offset))
    
    #print("Offset: {} | Before first access".format(offset))
    
    whiten_here = aux_info[2] < 0.5
    
    #print("Offset: {} | After first access".format(offset))
    
    #cache = tonumpyarray(mp_arr)
    
    #print("Offset: {} | After numpy".format(offset))
    
    #numpy_array = cache.reshape((int(aux_info[0]), int(aux_info[1])))
    
    #print("Offset: {} | After reshape: {}".format(offset, numpy_array.shape))
    
    #print("Offset: {} | Array: {}".format(offset, mp_arr.shape))
    
    numpy_array = mp_arr.reshape((int(aux_info[0]), int(aux_info[1])))
    #numpy_array = numpy_array.reshape((int(aux_info[0]), int(aux_info[1])))
    
    #print("Offset: {} | Numpy Array: {}".format(offset, numpy_array[0]))
    
    sample_list = []
    
    #print("Offset: {} | aux[0] = {}".format(offset, int(aux_info[0])))
    for i in range(int(aux_info[0])):
        #print("Offset: {} | i: {}".format(offset, i))
        dt = numpy_array[i][-2]
        epoch = numpy_array[i][-1]
        sample_rate = int(round(1.0 / dt))
        endpoint = int(len(numpy_array[i]) - 2 - offset * sample_rate)
        #print("Offset: {} | endpoint: {}".format(offset, endpoint))
        if whiten_here:
            #print("Offset: {} | in if".format(offset))
            #print("Offset: {} | Trying to access: {}".format(offset, (i, endpoint-int((64.0+aux_info[4])*sample_rate), endpoint)))
            #print("Offset: {} | Array: {}".format(offset, numpy_array))
            #print("aux_info[4]: {}".format(aux_info[4]))
            white = TimeSeries(numpy_array[i][endpoint-int((64.0 + aux_info[4])*sample_rate):endpoint], delta_t=dt, epoch=epoch).whiten(aux_info[3], aux_info[4], low_frequency_cutoff=20.0)
            #print("Offset: {} | after white".format(offset))
        else:
            #print("Offset: {} | in else".format(offset))
            white = TimeSeries(numpy_array[i][endpoint-int(64.*sample_rate):endpoint], delta_t=dt, epoch=epoch)
        sample_list.append(resample(white))
    
    ret = []
    for d in sample_list:
        ret += d
    
    #print("Will return now! | Offset: {}".format(offset))
    
    return(ret)

###############################################################################

def evaluate_ts(ts, net_path, time_step=0.25, preemptive_whiten=False, whiten_len=4., whiten_crop=4.):
    net = keras.models.load_model(net_path)
    
    if preemptive_whiten:
        for i in range(len(ts)):
            ts[i] = ts[i].whiten(whiten_len, whiten_crop, low_frequency_cutoff=20.0)
    
    mp_arr = mp.Array(c.c_double, len(ts) * (len(ts[0]) + 2))
    
    cache = tonumpyarray(mp_arr)
    
    numpy_array = cache.reshape((len(ts), len(ts[0]) + 2))
    
    for idx, d in enumerate(ts):
        numpy_array[idx][:len(d)] = d.data[:]
        numpy_array[idx][-2] = d.delta_t
        numpy_array[idx][-1] = d.start_time
    
    #print(numpy_array)
    
    aux_info = mp.Array(c.c_double, 5)
    
    aux_info[0] = len(ts)
    aux_info[1] = len(ts[0]) + 2
    aux_info[2] = 1 if preemptive_whiten else 0
    aux_info[3] = whiten_len
    aux_info[4] = whiten_crop
    
    time_shift_back = ts[0].duration - (64.0 if preemptive_whiten else (64.0+whiten_crop) )
    
    indexes = list(np.arange(time_shift_back, 0.0, -time_step))
    
    inp = []
    
    bar = progress_tracker(len(np.arange(time_shift_back, 0.0, -time_step)), name='Generating slices')
    
    with closing(mp.Pool(initializer=init, initargs=(mp_arr, aux_info))) as pool:
        #inp =  list(pool.imap(get_slice, np.arange(time_shift_back, 0.0, -time_step)))
        for idx, l in enumerate(pool.imap(get_slice, np.arange(time_shift_back, 0.0, -time_step))):
            inp.append(l)
            bar.iterate()
    pool.join()
    
    #print("Inp")
    
    #print(inp)
    
    inp = np.array(inp)

    inp = inp.transpose((1,0,2))
    
    real_inp = [np.zeros((2, inp.shape[1], inp.shape[2])) for i in range(14)]
    
    for i in range(14):
        real_inp[i][0] = inp[i]
        real_inp[i][1] = inp[i+14]
        real_inp[i] = real_inp[i].transpose(1,2,0)

    true_pred = net.predict(inp, verbose=1)
    
    snrs = list(true_pred[0].flatten())
    bools = [pt[0] for pt in true_pred[1]]

    snr_ts = TimeSeries(snrs, delta_t=time_step)
    bool_ts = TimeSeries(bools, delta_t=time_step)
    snr_ts.start_time = ts[0].start_time + (64.0 if preemptive_whiten else (64.0 + whiten_crop / 2.0))
    bool_ts.start_time = ts[0].start_time + (64.0 if preemptive_whiten else (64.0 + whiten_crop / 2.0))
    
    print(snr_ts.sample_times)
    
    return((snr_ts.copy(), bool_ts.copy()))
