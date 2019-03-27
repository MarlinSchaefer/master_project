import h5py
import numpy as np
import ast

def load_data(file_path):
    """Load data and labels for the training and testing set.
    
    Arguments
    ---------
    file_path : str
        Path to the file to load from. (Expected to exist)
    
    Returns
    -------
    tuple
        On the first level the tuple contains the training entries in the first
        and the testing entries in the second part of the tuple.
        On the second level the first entry is the data and the second entry
        the labels.
    """
    tr_data = load_training_data(file_path)
    tr_labels = load_training_labels(file_path)
    te_data = load_testing_data(file_path)
    te_labels = load_testing_labels(file_path)
    
    return(((tr_data, tr_labels), (te_data, te_labels)))

def load_calculated_snr(file_path):
    """Load the calculated SNR for all templates.
    
    Arguments
    ---------
    file_path : str
        Path to the file to load from. (Expected to exist)
    
    Returns
    -------
    tuple
        Tuple containing the SNR for the trainings set in the first and the
        ones for the testing set in the second entry.
    """
    with h5py.File(file_path, 'r') as data:
        ret = (data['training']['train_snr_calculated'].value, data['testing']['test_snr_calculated'].value)
    return(ret)

def load_training_data(file_path):
    """Load just the training data.
    
    Arguments
    ---------
    file_path : str
        Path to the file to load from. (Expected to exist)
    
    Returns
    -------
    numpy array
        The numpy array is of the shape:
        (num of templates, num of samples per template, num of detectors)
    """
    with h5py.File(file_path, 'r') as inp:
        ret = inp['training']['train_data'].value
    return(ret)

def load_training_labels(file_path):
    """Load just the training labels.
    
    Arguments
    ---------
    file_path : str
        Path to the file to load from. (Expected to exist)
    
    Returns
    -------
    numpy array
        The numpy array is of the shape:
        (num of templates, 1)
    """
    with h5py.File(file_path, 'r') as inp:
        ret = inp['training']['train_labels'].value
    return(ret)

def load_training_calculated_snr(file_path):
    """Load just the calculated SNR of the training set.
    
    Arguments
    ---------
    file_path : str
        Path to the file to load from. (Expected to exist)
    
    Returns
    -------
    numpy array
        The numpy array is of the shape:
        (num of templates, )
    """
    with h5py.File(file_path, 'r') as inp:
        ret = inp['training']['train_snr_calculated'].value
    return(ret)

def load_training_wave_parameters(file_path):
    """Load just the wave parameters of the training set.
    
    Arguments
    ---------
    file_path : str
        Path to the file to load from. (Expected to exist)
    
    Returns
    -------
    list
        A list containing dictionaries giving the wave parameters for each
        template.
    """
    print(file_path)
    with h5py.File(file_path, 'r') as inp:
        ret = inp['training']['parameters']['wav_parameters'].value
    
    #BUG: Not all dictionaries seem to be complete!
    ret = [ast.literal_eval(s) for s in ret]
            
    return(ret)

def load_training_external_parameters(file_path):
    """Load just the wave parameters of the training set.
    
    Arguments
    ---------
    file_path : str
        Path to the file to load from. (Expected to exist)
    
    Returns
    -------
    list
        A list containing dictionaries giving the wave parameters for each
        template.
    """
    with h5py.File(file_path, 'r') as inp:
        ret = inp['training']['parameters']['ext_parameters'].value
    
    #BUG: Not all dictionaries seem to be complete!
    ret = [ast.literal_eval(s) for s in ret]
    
    return(ret)

def load_testing_data(file_path):
    """Load just the testing data.
    
    Arguments
    ---------
    file_path : str
        Path to the file to load from. (Expected to exist)
    
    Returns
    -------
    numpy array
        The numpy array is of the shape:
        (num of templates, num of samples per template, num of detectors)
    """
    with h5py.File(file_path, 'r') as inp:
        ret = inp['testing']['test_data'].value
    return(ret)

def load_testing_labels(file_path):
    """Load just the testing labels.
    
    Arguments
    ---------
    file_path : str
        Path to the file to load from. (Expected to exist)
    
    Returns
    -------
    numpy array
        The numpy array is of the shape:
        (num of templates, 1)
    """
    with h5py.File(file_path, 'r') as inp:
        ret = inp['testing']['test_labels'].value
    return(ret)

def load_testing_calculated_snr(file_path):
    """Load just the calculated SNR of the testing set.
    
    Arguments
    ---------
    file_path : str
        Path to the file to load from. (Expected to exist)
    
    Returns
    -------
    numpy array
        The numpy array is of the shape:
        (num of templates, )
    """
    with h5py.File(file_path, 'r') as inp:
        ret = inp['testing']['test_snr_calculated'].value
    return(ret)

def load_testing_wave_parameters(file_path):
    """Load just the wave parameters of the testing set.
    
    Arguments
    ---------
    file_path : str
        Path to the file to load from. (Expected to exist)
    
    Returns
    -------
    list
        A list containing dictionaries giving the wave parameters for each
        template.
    """
    with h5py.File(file_path, 'r') as inp:
        ret = inp['testing']['parameters']['wav_parameters'].value
    
    #BUG: Not all dictionaries seem to be complete!
    ret = [ast.literal_eval(s) for s in ret]
    
    return(ret)

def load_testing_external_parameters(file_path):
    """Load just the wave parameters of the testing set.
    
    Arguments
    ---------
    file_path : str
        Path to the file to load from. (Expected to exist)
    
    Returns
    -------
    list
        A list containing dictionaries giving the wave parameters for each
        template.
    """
    with h5py.File(file_path, 'r') as inp:
        ret = inp['testing']['parameters']['ext_parameters'].value
    
    #BUG: Not all dictionaries seem to be complete!
    ret = [ast.literal_eval(s) for s in ret]
    
    return(ret)

def load_psd_data(file_path):
    """Load the data for the PSD.
    
    Arguments
    ---------
    file_path : str
        Path to the file to load from. (Expected to exist)
    
    Returns
    -------
    numpy array
        numpy array of the shape:
        (num of samples, )
    """
    with h5py.File(file_path, 'r') as inp:
        ret = inp['psd']['data'].value
    return(ret)

def load_psd_delta_f(file_path):
    """Load just the delta f for the PSD.
    
    Arguments
    ---------
    file_path : str
        Path to the file to load from. (Expected to exist)
    
    Returns
    -------
    numpy float64
    """
    with h5py.File(file_path, 'r') as inp:
        ret = inp['psd']['delta_f'].value
    return(ret)

def load_parameter_space(file_path):
    """Load the parameter space for the templates.
    
    Arguments
    ---------
    file_path : str
        Path to the file to load from. (Expected to exist)
    
    Returns
    -------
    dict
        Dictionary containing all the parameters and their ranges.
    """
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

def get_number_training_samples(file_path):
    with h5py.File(file_path, 'r') as data:
        n = len(data['training']['train_data'])
    return(n)

def get_number_testing_samples(file_path):
    with h5py.File(file_path, 'r') as data:
        n = len(data['testing']['test_data'])
    return(n)
