import h5py
import numpy as np

FILE_NAME = "template_bank_full"

def load_data():
    data = h5py.File(FILE_NAME + ".hdf5", 'r')
    
    train_data = data['training']['train_data'][:]
    train_labels = data['training']['train_labels'][:]
    
    test_data = data['testing']['test_data'][:]
    test_labels = data['testing']['test_labels'][:]
    
    return(((train_data, train_labels), (test_data, test_labels)))
