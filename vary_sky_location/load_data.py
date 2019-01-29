import h5py
import numpy as np

def load_data(file_path):
    data = h5py.File(file_path, "r")
    
    train_data = data['training']['train_data'][:]
    train_labels = data['training']['train_labels'][:]
    
    test_data = data['testing']['test_data'][:]
    test_labels = data['testing']['test_labels'][:]
    
    data.close()
    
    return(((train_data, train_labels), (test_data, test_labels)))

def load_wiki(file_path):
    ret_dict = {}
    
    data = h5py.File(file_path, "r")
    
    for key, val in data['parameter_space'].items():
        ret_dict[str(key)] = val.value
    
    data.close()
    
    return(ret_dict)

def load_calculated_snr(file_path):
    with h5py.File(file_path, 'r') as data:
        ret = (data['training']['train_snr_calculated'].value, data['testing']['test_snr_calculated'].value)
    
    return(ret)

def load_training_data(file_path):
    with h5py.File(file_path, 'r') as inp:
        ret = inp['training']['train_data'].value
    return(ret)

def load_training_labels(file_path):
    with h5py.File(file_path, 'r') as inp:
        ret = inp['training']['train_labels'].value
    return(ret)

def load_training_calculated_snr(file_path):
    with h5py.File(file_path, 'r') as inp:
        ret = inp['training']['train_snr_calculated'].value
    return(ret)

def load_training_wave_parameters(file_path):
    with h5py.File(file_path, 'r') as inp:
        ret = inp['training']['parameters']['wav_parameters'].value
    return(ret)

def load_training_external_parameters(file_path):
    with h5py.File(file_path, 'r') as inp:
        ret = inp['training']['parameters']['ext_parameters'].value
    return(ret)

def load_testing_data(file_path):
    with h5py.File(file_path, 'r') as inp:
        ret = inp['testing']['test_data'].value
    return(ret)

def load_testing_labels(file_path):
    with h5py.File(file_path, 'r') as inp:
        ret = inp['testing']['test_labels'].value
    return(ret)

def load_testing_calculated_snr(file_path):
    with h5py.File(file_path, 'r') as inp:
        ret = inp['testing']['test_snr_calculated'].value
    return(ret)

def load_testing_wave_parameters(file_path):
    with h5py.File(file_path, 'r') as inp:
        ret = inp['testing']['parameters']['wav_parameters'].value
    return(ret)

def load_testing_external_parameters(file_path):
    with h5py.File(file_path, 'r') as inp:
        ret = inp['testing']['parameters']['ext_parameters'].value
    return(ret)

def load_psd_data(file_path):
    with h5py.File(file_path, 'r') as inp:
        ret = inp['psd']['data'].value
    return(ret)

def load_psd_delta_f(file_path):
    with h5py.File(file_path, 'r') as inp:
        ret = inp['psd']['delta_f'].value
    return(ret)

def load_parameter_space(file_path):
    with h5py.File(file_path, 'r') as inp:
        ret = {}
        tmp = inp['parameter_space']
        for key in tmp.keys():
            ret[str(key)] = tmp[key].value
    return(ret)

def load_full_data(file_path):
    """Loads and returns all the data in the file.
    
    Arguments
    ---------
    file_path : str
        A string containing the path to the .hf5 to be read.
    
    Returns
    -------
    tuple
        Returns a tuple of tuples. Where the first entry in the toplevel tuple
        contains all data for the training set, the second all data for the
        testing set, the third all data for the PSD and the fourth all
        parameters that were used to create the data. For details on the
        structure of each of these pieces of data see the documentation of the
        according loading funtion within this module
    """
    tr_data = load_training_data(file_path)
    tr_labels = load_training_labels(file_path)
    tr_snr = load_training_calculated_snr(file_path)
    tr_wav = load_training_wave_parameters(file_path)
    tr_ext = load_training_external_parameters(file_path)
    
    te_data = load_testing_data(file_path)
    te_labels = load_testing_labels(file_path)
    te_snr = load_testing_calculated_snr(file_path)
    te_wav = load_testing_wave_parameters(file_path)
    te_ext = load_testing_external_parameters(file_path)
    
    psd_dat = load_psd_data(file_path)
    df = load_psd_delta_f(file_path)
    
    parameter_space = load_parameter_space(file_path)
    
    return(((tr_data, tr_labels, tr_snr, tr_wav, tr_ext), (te_data, te_labels, te_snr, te_wav, te_ext), (psd_dat, df), parameter_space))
