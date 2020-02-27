import h5py
import os
from aux_functions import get_store_path
import numpy as np

def get_dir_path():
    return os.path.join(get_store_path(), 'final_network_retrain_dev21_3')

def get_collected_results_path():
    return os.path.join(get_dir_path(), 'collected_results.hf5')

def get_percentage():
    target = 22 * (201 - 3) / 2
    curr = 0
    with h5py.File(get_collected_results_path(), 'r') as f:
        for highLevel in np.arange(3, 202, 2):
            for lowLevel in np.arange(22):
                if str(highLevel) in f.keys():
                    if str(lowLevel) in f[str(highLevel)].keys():
                        curr += 1
    return float(curr) / target

def main():
    collected_results_path = get_collected_results_path()
    with h5py.File(collected_results_path, 'a') as f:
        existing_keys = {}
        for key in f.keys():
            highLevel = int(key)
            existing_keys[highLevel] = []
            for k in f[key].keys():
                lowLevel = int(k)
                existing_keys[highLevel].append(lowLevel)
        
        for highLevel in np.arange(3, 202, 2, dtype=int):
            for lowLevel in np.arange(22, dtype=int):
                file_path = os.path.join(get_dir_path(),
                                         '{}_{}'.format(highLevel, lowLevel),
                                         'result_{}_{}.hf5'.format(highLevel, lowLevel))
                if os.path.isfile(file_path):
                    if highLevel not in existing_keys:
                        existing_keys[highLevel] = []
                        f.create_group(str(highLevel))
                    if lowLevel not in existing_keys[highLevel]:
                        with h5py.File(file_path, 'r') as read:
                            group = f[str(highLevel)].create_group(str(lowLevel))
                            
                            ts = group.create_group('TimeSeries')
            
                            ts.create_dataset('net_path', data=read['net_path'][()])
                            
                            ts_snr = ts.create_group('snrTimeSeries')
                            ts_snr.create_dataset('data', data=read['snrTimeSeries/data'][:])
                            ts_snr.create_dataset('sample_times', data=read['snrTimeSeries/sample_times'][:])
                            ts_snr.create_dataset('delta_t', data=read['snrTimeSeries/delta_t'][()])
                            ts_snr.create_dataset('epoch', data=read['snrTimeSeries/epoch'][()])
                            
                            ts_bool = ts.create_group('p-valueTimeSeries')
                            ts_bool.create_dataset('data', data=read['p-valueTimeSeries/data'][:])
                            ts_bool.create_dataset('sample_times', data=read['p-valueTimeSeries/sample_times'][:])
                            ts_bool.create_dataset('delta_t', data=read['p-valueTimeSeries/delta_t'][()])
                            ts_bool.create_dataset('epoch', data=read['p-valueTimeSeries/epoch'][()])
    
    print("File complete to {0:.2f}%.".format(100 * get_percentage()))
    return

if __name__ == '__main__':
    main()
