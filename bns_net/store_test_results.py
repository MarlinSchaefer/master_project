import keras
import numpy as np
import h5py
import generator as g
import gc
import os

def store_test_results(net, dobj, store_path, batch_size=32):
    res = net.predict_generator(g.DataGeneratorMultInput(dobj.loaded_test_data, dobj.loaded_test_labels, batch_size=batch_size, shuffle=False), verbose=1)
    
    print(res)
    
    if type(res) == list:
        shape = [0, 0]
        shape[0] = len(res[0])
        for con in res:
            shape[1] += con.shape[-1]
        st = np.empty(shape)
        j = 0
        for con in res:
            for i in range(len(con)):
                for k in range(len(con[i])):
                    st[i][j+k] = con[i][k]
            j += len(con[0])
        
        with h5py.File(store_path, 'w') as FILE:
            FILE.create_dataset('data', data=st)
    else:
        with h5py.File(store_path, 'w') as FILE:
            FILE.create_dataset('data', data=res)
    
    print("Stored data at: {}\n".format(store_path))
    
    return(store_path)

def join_test_results(files, return_path, delete_copied_files=False):
    if not type(files) == list or type(files) == str:
        raise ValueError("The argument of 'join_test_result' has to be either a string or a list of strings.")
    
    if type(files) == str:
        return(files)
    
    shapes = {}
    dtypes = {}
    
    for f in files:
        try:
            with h5py.File(f, 'r') as FILE:
                shapes[f] = FILE['data'].shape
                dtypes[f] = FILE['data'].dtype
        except:
            pass
    
    if len(shapes) == 0:
        raise IOError('There were no files of the correct format provided.')
    
    ref_shape = shapes.values()[0]
    
    length_total = 0
    
    for v in shapes.values():
        if v[1:] == ref_shape[1:]:
            length_total += v[0]
        else:
            raise ValueError('Not all data has the same shape after axis 0.')
    
    with h5py.File(return_path, 'w') as ret_file:
        ret_file.create_dataset('data', shape=tuple([length_total] + list(ref_shape[1:])), dtype=dtypes.values()[0])
        last_ind = 0
        for f in shapes.keys():
            with h5py.File(f, 'r') as read_file:
                ret_file['data'][last_ind:last_ind+shapes[f][0]] = read_file['data'][:]
                last_ind = last_ind+shapes[f][0]
            if delete_copied_files:
                os.remove(f)
    
    return(return_path)

#def store_full_results(net, dobj, store_path, batch_size=32):
    #dobj.join_formatted('training', 'train_data', dobj.loaded_train_data, dobj.loaded_test_data)
    #dobj.unload_type('testing', 'test_data')
    #gc.collect()
    #dobj.join_formatted('training', 'train_labels', dobj.loaded_train_labels, dobj.loaded_test_labels)
    #dobj.unload_type('testing', 'test_labels')
    #gc.collect()
    
    #res = net.predict_generator(g.DataGeneratorMultInput(dobj.loaded_train_data, dobj.loaded_train_labels, batch_size=batch_size, shuffle=False), verbose=1)
    
    #if type(res) == list:
        #shape = [0, 0]
        #shape[0] = len(res[0])
        #for con in res:
            #shape[1] += con.shape[-1]
        #st = np.empty(shape)
        #j = 0
        #for con in res:
            #for i in range(len(con)):
                #for k in range(len(con[i])):
                    #st[i][j+k] = con[i][k]
            #j += len(con[0])
        
        #with h5py.File(store_path, 'w') as FILE:
            #FILE.create_dataset('data', data=st)
    #else:
        #with h5py.File(store_path, 'w') as FILE:
            #FILE.create_dataset('data', data=res)
    
    #print("Stored data at: {}\n".format(store_path))
    
    #dobj.unload_all()
    #gc.collect()
