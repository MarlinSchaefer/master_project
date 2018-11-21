import h5py
import numpy as np

FILE_NAME = "template_bank_full"

def load_data_Conv1D():
    data = h5py.File(FILE_NAME + ".hdf5", 'r')
    
    train_data = data['training']['train_data'][:]
    train_labels = data['training']['train_labels'][:]
    
    test_data = data['testing']['test_data'][:]
    test_labels = data['testing']['test_labels'][:]
    
    #"""
    train_data = train_data.reshape((len(train_data),len(train_data[0]),1))
    #train_labels = train_labels.reshape((len(train_labels),len(train_labels[0]),1))
    
    test_data = test_data.reshape((len(test_data),len(test_data[0]),1))
    #test_labels = test_labels.reshape((len(test_labels),len(test_labels[0]),1))
    #"""
    
    return(((train_data, train_labels), (test_data, test_labels)))

def load_data():
    data = h5py.File(FILE_NAME + ".hdf5", 'r')
    
    train_data = data['training']['train_data'][:]
    train_labels = data['training']['train_labels'][:]
    
    test_data = data['testing']['test_data'][:]
    test_labels = data['testing']['test_labels'][:]
    
    return(((train_data, train_labels), (test_data, test_labels)))
