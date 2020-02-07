import h5py
import numpy as np
from aux_functions import get_store_path
import os

def get_results_path():
    return os.path.join(get_store_path(), 'long_data_2', 'results')

def getRunStatsPath(highLevel):
    return os.path.join(get_store_path(), 'long_data_2', 'run_stats_' + str(highLevel) + '.hdf')

def main():
    additive_time = float(4096 * 22)
    injectionTimes = np.array([])
    cphase = np.array([])
    dec = np.array([])
    dist = np.array([])
    inc = np.array([])
    mass1 = np.array([])
    mass2 = np.array([])
    pol = np.array([])
    ra = np.array([])
    
    for idx, highLevel in enumerate(np.arange(3, 202, 2)):
        with h5py.File(getRunStatsPath(highLevel), 'r') as f:
            injectionTimes = np.concatenate([injectionTimes, idx * additive_time + f['times'][:]])
            cphase = np.concatenate([cphase, f['cphase'][:]])
            dec = np.concatenate([dec, f['dec'][:]])
            dist = np.concatenate([dist, f['dist'][:]])
            inc = np.concatenate([inc, f['inc'][:]])
            mass1 = np.concatenate([mass1, f['mass1'][:]])
            mass2 = np.concatenate([mass2, f['mass2'][:]])
            pol = np.concatenate([pol, f['pol'][:]])
            ra = np.concatenate([ra, f['ra'][:]])
    
    with h5py.File(os.path.join(get_results_path(), 'collected_stats.hf5'), 'w') as f:
        f.create_dataset('times', data=injectionTimes)
        f.create_dataset('cphase', data=cphase)
        f.create_dataset('dec', data=dec)
        f.create_dataset('dist', data=dist)
        f.create_dataset('inc', data=inc)
        f.create_dataset('mass1', data=mass1)
        f.create_dataset('mass2', data=mass2)
        f.create_dataset('pol', data=pol)
        f.create_dataset('ra', data=ra)
    
    return

main()
