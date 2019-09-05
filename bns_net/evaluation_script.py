import keras
import numpy as np
import h5py
import sys
import os
from aux_functions import get_store_path
from generator import generatorFromTimeSeriesReducedSplit
from detection_pipeline import evaluate_ts_from_generator
from pycbc.types import TimeSeries

#Command line input, should be integer
data_num = int(sys.argv[1])
#Name of directory
dir_name = 'evaluation_results'
#Name of data file including extension
data_name = 'data-1.hdf'
#Name of file that stores time series results
ts_name = 'resulting_ts_' + str(data_num) + '.hf5'
#Name of file containing triggers
trigger_name = 'triggers_' + str(data_num) + '.hf5'
#The path to the directory, where results are stored
dir_path = os.path.join(get_store_path(), dir_name)
#Path to the data that should be evaluated
#data_path = os.path.join('/home/ahnitz/projects/ml_mock_data', data_name)
data_path = os.path.join(get_store_path(), data_name)
#Path where resulting time series are stored
ts_path = os.path.join(dir_path, ts_name)
#Path to the triggers of the data
trigger_path = os.path.join(dir_path, trigger_name)
#Path to the network
#net_path = '/home/marlin.schaefer/master_project/bns_net/saves/tcn_collect_inception_res_net_rev_6_248201905643/tcn_collect_inception_res_net_rev_6_epoch_21.hf5'
net_path = os.path.join(get_store_path(), 'tcn_collect_inception_res_net_rev_6_248201905643', 'tcn_collect_inception_res_net_rev_6_epoch_21.hf5')

#Thresholds for the network
snr_threshold = 6.7010087966918945
bool_threshold = 0.408351868391037

#Check if directory exists, if not create it
try:
    os.mkdir(dir_path)
except OSError:
    pass

#Load data into memory and convert it to time series
with h5py.File(data_path, 'r') as FILE:
    data = [TimeSeries(FILE['L1'][str(data_num)][:], delta_t=1.0/4096),
            TimeSeries(FILE['H1'][str(data_num)][:], delta_t=1.0/4096)]

#Evaluate data
snr_ts, bool_ts = evaluate_ts_from_generator(data, net_path, generatorFromTimeSeriesReducedSplit)

#Store time series to HDF-file
with h5py.File(ts_path, 'w') as FILE:
    FILE.create_dataset('net_path', data=np.str(net_path))
    FILE.create_dataset('threshold_snr', data=np.float(snr_threshold))
    FILE.create_dataset('threshold_p-value', data=np.float(bool_threshold))
    
    hf5_snr_ts = FILE.create_group('snrTimeSeries')
    hf5_snr_ts.create_dataset('data', data=snr_ts.data[:])
    hf5_snr_ts.create_dataset('sample_times', data=snr_ts.sample_times[:])
    hf5_snr_ts.create_dataset('delta_t', data=snr_ts.delta_t)
    hf5_snr_ts.create_dataset('epoch', data=np.float(snr_ts._epoch))
    
    hf5_bool_ts = FILE.create_group('p-valueTimeSeries')
    hf5_bool_ts.create_dataset('data', data=bool_ts.data[:])
    hf5_bool_ts.create_dataset('sample_times', data=bool_ts.sample_times[:])
    hf5_bool_ts.create_dataset('delta_t', data=bool_ts.delta_t)
    hf5_bool_ts.create_dataset('epoch', data=np.float(bool_ts._epoch))

#Compute SNR triggers
bar = progress_tracker(len(snr_ts), name='Generating SNR triggers')
snr_val = []
snr_time = []
for i in range(len(snr_ts)):
    if snr_ts[i] > snr_threshold:
        snr_val.append(snr_ts[i])
        snr_time.append(snr_ts.sample_times[i])
    bar.iterate()

snr_val = np.array(snr_val)
snr_time = np.array(snr_time)

#Compute p-value triggers
bar = progress_tracker(len(bool_ts), name='Generating p-value triggers')
bool_val = []
bool_time = []
for i in range(len(bool_ts)):
    if bool_ts[i] > bool_threshold:
        bool_val.append(bool_ts[i])
        bool_time.append(bool_ts.sample_times[i])
    bar.iterate()

bool_val = np.array(bool_val)
bool_time = np.array(bool_time)

#Compute combined triggers
bar = progress_tracker(len(snr_time), name='Generating combined triggers')
combined_snr_val = []
combined_bool_val = []
combined_trigger_times = []
for i in range(len(snr_time)):
    idx = np.where(bool_time == snr_time[i])[0]
    if len(idx) > 0:
        combined_snr_val.append(snr_val[i])
        combined_bool_val.append(bool_val[idx[0]])
        combined_trigger_times.append(snr_time[i])
    bar.iterate()

combined_snr_val = np.array(combined_snr_val)
combined_bool_val = np.array(combined_bool_val)
combined_trigger_times = np.array(combined_trigger_times)

#Store triggers
with h5py.File(trigger_path, 'w') as FILE:
    hf5_snr_triggers = FILE.create_group('snrTriggers')
    hf5_snr_triggers.create_dataset('triggerTimes', data=snr_time)
    hf5_snr_triggers.create_dataset('triggerValues', data=snr_val)
    
    hf5_bool_triggers = FILE.create_group('p-valueTriggers')
    hf5_bool_triggers.create_dataset('triggerTimes', data=bool_time)
    hf5_bool_triggers.create_dataset('triggerValues', data=bool_val)
    
    hf5_combined_triggers = FILE.create_group('combinedTriggers')
    hf5_combined_triggers.create_dataset('triggerTimes', data=combined_trigger_times)
    hf5_combined_triggers.create_dataset('snr-values', data=combined_snr_val)
    hf5_combined_triggers.create_dataset('p-values', data=combined_bool_val)
