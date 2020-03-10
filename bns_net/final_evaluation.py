#!/usr/bin/env python

import os
import sys
from aux_functions import get_store_path
import shutil
import keras
import collect_triggers_adjust as cta
import numpy as np
import h5py
import generator
from pycbc.types import TimeSeries
import detection_pipeline as dp

#Global definitions
PARENT_DIR_NAME = 'final_network_retrain_dev21_3'
NET_FILE = 'final_network_retrain_dev21_3_epoch_21.hf5'
LL = 0
HL = 0
WINDOW_TIME_COMPENSATION = 67.5 #Time duration of window - 4.5 for cropping and mean position of peak amplitude in training set

def set_levels(highLevel, lowLevel):
    global LL
    global HL
    LL = lowLevel
    HL = highLevel
    return

def create_directory(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return

def get_parent_dir_path():
    return os.path.join(get_store_path(), PARENT_DIR_NAME)

def copy_necessary(dir_path):
    net_path = os.path.join(dir_path, NET_FILE)
    if not os.path.isfile(net_path):
        shutil.copy(os.path.join(get_parent_dir_path(), NET_FILE), net_path)

def get_dir_path():
    data_dir_name = str(HL) + '_' + str(LL)
    return os.path.join(get_parent_dir_path(), data_dir_name)

def get_result_path():
    #file_name = 'result_' + str(HL) + '_' + str(LL) + '.hf5'
    file_name = 'bbh_result_' + str(HL) + '_' + str(LL) + '.hf5'
    return os.path.join(get_dir_path(), file_name)

def get_data(file_path):
    #with h5py.File(file_path, 'r') as f:
        #data = [TimeSeries(f['L1'][()], delta_t=1./4096),
                #TimeSeries(f['H1'][()], delta_t=1./4096)]
    #return data
    with h5py.File(file_path, 'r') as f:
        data = [TimeSeries(f['L1/0'][()], delta_t=1./4096),
                TimeSeries(f['H1/0'][()], delta_t=1./4096)]
    return data

def get_data_dir():
    #return os.path.join(get_store_path(), 'long_data_2')
    return os.path.join(get_store_path(), 'bbhTest')

def get_file_path():
    #return os.path.join(get_data_dir(), 'data-' + str(HL) + '_part_' + str(LL) + '.hf5')
    return os.path.join(get_data_dir(), 'data-' + str(HL) + str(LL) + '.hdf')

def call(highLevel, lowLevel):
    #Initial setup
    set_levels(highLevel, lowLevel)
    dir_path = get_dir_path()
    parent_dir = get_parent_dir_path()
    create_directory(dir_path)
    net_path = os.path.join(dir_path, NET_FILE)
    result_path = get_result_path()
    
    #Copy the network over
    copy_necessary(dir_path)
    
    #Set data
    boundaries = cta.boundaries(highLevel, lowLevel)
    data = get_data(get_file_path())
    
    #Call pipeline
    gen = generator.generatorFromTimeSeriesReducedSplitRemoveLast
    snr_ts, bool_ts = dp.evaluate_ts_from_generator(data, net_path, gen)
    
    #Store results
    with h5py.File(result_path, 'w') as f:
        f.create_dataset('net_path', data=np.str(net_path))
        
        hf5_snr_ts = f.create_group('snrTimeSeries')
        hf5_snr_ts.create_dataset('data', data=snr_ts.data[:])
        hf5_snr_ts.create_dataset('sample_times', data=snr_ts.sample_times[:] + boundaries[0] + WINDOW_TIME_COMPENSATION)
        hf5_snr_ts.create_dataset('delta_t', data=snr_ts.delta_t)
        hf5_snr_ts.create_dataset('epoch', data=np.float(snr_ts._epoch) + boundaries[0] + WINDOW_TIME_COMPENSATION)

        hf5_bool_ts = f.create_group('p-valueTimeSeries')
        hf5_bool_ts.create_dataset('data', data=bool_ts.data[:])
        hf5_bool_ts.create_dataset('sample_times', data=bool_ts.sample_times[:] + boundaries[0] + WINDOW_TIME_COMPENSATION)
        hf5_bool_ts.create_dataset('delta_t', data=bool_ts.delta_t)
        hf5_bool_ts.create_dataset('epoch', data=np.float(bool_ts._epoch) + boundaries[0] + WINDOW_TIME_COMPENSATION)
    
    return result_path

def main():
    highLevel = int(sys.argv[1])
    lowLevel = int(sys.argv[2])
    call(highLevel, lowLevel)
    return

if __name__ == "__main__":
    main()
