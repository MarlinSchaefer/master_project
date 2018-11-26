import h5py
import numpy as np

FILE_NAME = "template_bank_full"

def load_data_Conv1D(name=FILE_NAME):
    data = h5py.File(name + ".hdf5", 'r')
    
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

def load_data(name=FILE_NAME):
    data = h5py.File(name + ".hdf5", 'r')
    
    train_data = data['training']['train_data'][:]
    train_labels = data['training']['train_labels'][:]
    
    test_data = data['testing']['test_data'][:]
    test_labels = data['testing']['test_labels'][:]
    
    return(((train_data, train_labels), (test_data, test_labels)))

def fix_wrong_indexed(name=FILE_NAME):
    data = h5py.File(name + '.hdf5', 'r')
    
    tot_data = np.concatenate((data['training']['train_data'][:], data['testing']['test_data'][:]))
    tot_labels = np.concatenate((data['training']['train_labels'][:], data['testing']['test_labels'][:]))
    
    data.close()
    
    split_index = int(round(0.7 * len(tot_data)))
    
    train_data = tot_data[:split_index]
    train_labels = tot_labels[:split_index]
    
    test_data = tot_data[split_index:]
    test_labels = tot_labels[split_index:]
    
    output = h5py.File(name + '_new.hdf5', 'w')
    
    training = output.create_group('training')
    training.create_dataset('train_data', data=train_data, dtype='f')
    training.create_dataset('train_labels', data=train_labels, dtype='f')
    
    testing = output.create_group('testing')
    testing.create_dataset('test_data', data=test_data)
    testing.create_dataset('test_labels', data=test_labels)
    
    output.close()
