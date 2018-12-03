from pycbc.waveform import get_td_waveform
from multiprocessing import Pool
import numpy as np
from random import seed, random

#TODO
"""
-How do I call this script using the multiprocessing module when it has to be wrapped in if __name__ == "__main__"
-Implement random distribution for an array
"""

def payload()

def create_file(**kwargs):
    #Properties for the generating program
    opt_arg['gw_prob'] = 1.0
    opt_arg['snr'] = [1.0, 12.0]
    opt_arg['resample_delta_t'] = 1.0 / 1024
    opt_arg['t_len'] = 64.0
    opt_arg['resample_t_len'] = 4.0
    opt_arg['num_of_templates'] = 20000
    opt_arg['random_starting_time'] = True
    opt_arg['seed'] = 12345
    
    for key in opt_arg.keys():
        if key is in kwargs:
            opt_arg[key] = kwargs.get(key)
            del kwargs[key]
    
    #Properties for the waveform itself
    wav_arg['approximant'] = "SEOBNRv4_opt"
    wav_arg['mass1'] = 30.0
    wav_arg['mass2'] = 30.0
    wav_arg['delta_t'] = 1.0 / 4096
    wav_arg['f_lower'] = 20.0
    wav_arg['phase'] = [0., np.pi]
    
    for key in wav_arg.keys():
        if key is in kwargs:
            wav_arg[key] = kwargs.get(key)
            del kwargs[key]
    
    for key, val in kwargs.items():
        if not key == 'mode_array':
            if type(val) == list:
                kwargs[key] = [min(val), max(val)]
