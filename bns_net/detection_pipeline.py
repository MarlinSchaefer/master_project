import keras
import numpy as np
from pycbc.types.timeseries import TimeSeries
import os
from pycbc.filter import resample_to_delta_t
from progress_bar import progress_tracker

#TODO: Implement this. Make it act on data of the correct length
def format_data(data):
    """Correctly formats the data for the network.
    
    Arguments
    ---------
    data : list
        A list containing the raw strain data from the different detectors
    
    Returns
    -------
    numpy array
        numpy array containg the data in the correct shape.
    """
    
    #Formats data to suite
    
    print("Length of data: {}".format(len(data)))
    
    dat_1s = []
    
    np.array([ret])
    
    return(ret)

#TODO: Make this run in parallel
def get_slices(data, shift, window_length):
    ret = []
    
    #inverse delta t
    idt = [4096, 2048, 1024, 512, 256, 128, 64]
    
    window_factor = [int(4096.0 / pt) for pt in idt]
    
    window_sizes = [window_length * wf for wf in window_factor]
    
    max_window_size = max(window_sizes)
    
    num_of_shifts = (len(data[0]) - max_window_size) // shift
    
    bar = progress_tracker(num_of_shifts, name='Generating slices')
    
    for i in range(num_of_shifts):
        total = []
        for idx, ws in enumerate(window_sizes):
            for dat in data:
                resampled = resample_to_delta_t(dat[i*shift+(max_window_size-ws):i*shift+max_window_size], 1.0 / idt[idx])
                total.append(resampled)
        ret.append(np.array(total).transpose())
        
        bar.iterate()
    
    del data
    return(np.array(ret))

def get_snr_timeseries(net_path, data, time_shift=0.5, window_length=4096):
    """Return a TimeSeries with the datapoints containing the SNR as estimated
    by the given network.
    
    Arguments
    ---------
    net_path : str
        Full path to the network. The function expects a .hf5 file at that
        location
    
    data : list
        A list containing the raw strain data from the different detectors.
        This is expected to consist of TimeSeries objects that have already
        been aligned such that their epochs are the same.
    
    time_shift : float
        How much time (in seconds) should pass between SNR calculations. (Time 
        by which the analyzing window will be shifted.)
    
    window_length : int
        How many samples should be passed to the network. (This will be
        dictated by the networks input layer)
    
    Returns
    -------
    TimeSeries
        A time series containing a value for the SNR at every sampled point.
        The epoch will be the same as for the data provided.
    """
    
    #Sanity checks
    if not net_path[-4:] == '.hf5':
        raise ValueError('%s is not in the correct format. Please provide a path .hf5 file containing the network.' % net_path)
    
    if not isinstance(data, list):
        raise ValueError("The data argument needs to be of type 'list' and contain TimeSeries objects.")
    
    for dat in data:
        if not isinstance(dat, TimeSeries):
            raise ValueError("The data argument needs to be of type 'list' and contain TimeSeries objects.")
        if not dat._epoch == data[0]._epoch:
            continue
            #raise ValueError("The TimeSeries objects provided need to have matching epochs.")
        if not dat.delta_t == data[0].delta_t:
            raise ValueError("The TimeSeries need to have matching delta_t.")
    
    #Chop data into pieces of correct length to feed them to the network. This way memory should be minimized.
    
    #Find the maximum length
    len_data = min([len(dat) for dat in data])
    
    data = [dat[:len_data].whiten(4, 4) for dat in data]
    
    DELTA_T = data[0].delta_t
    shift = int(time_shift / DELTA_T)
    
    slices = get_slices(data, shift, window_length)
    
    return_time_series = TimeSeries(np.zeros(len(slices)), delta_t=time_shift, epoch=data[0].start_time)
    
    net = keras.models.load_model(net_path)
    
    res = net.predict(slices)
    
    print(res)
    
    for i, r in enumerate(res[0]):
        print(r.flatten())
        return_time_series[i] = r.flatten()[0]
    
    return(return_time_series)
            
