import h5py
import numpy as np

def load_data(file_path):
    data = h5py.File(file_path, "r")
    
    train_data = data['training']['train_data'][:]
    train_labels = data['training']['train_labels'][:]
    
    test_data = data['testing']['test_data'][:]
    test_labels = data['testing']['test_labels'][:]
    
    return(((train_data, train_labels), (test_data, test_labels)))
