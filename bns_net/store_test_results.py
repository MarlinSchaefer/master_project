import keras
import numpy as np
import h5py
import generator as g
import gc

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
    
def store_full_results(net, dobj, store_path, batch_size=32):
    dobj.join_formatted('training', 'train_data', dobj.loaded_train_data, dobj.loaded_test_data)
    dobj.unload_type('testing', 'test_data')
    gc.collect()
    dobj.join_formatted('training', 'train_labels', dobj.loaded_train_labels, dobj.loaded_test_labels)
    dobj.unload_type('testing', 'test_labels')
    gc.collect()
    
    res = net.predict_generator(g.DataGeneratorMultInput(dobj.loaded_train_data, dobj.loaded_train_labels, batch_size=batch_size, shuffle=False), verbose=1)
    
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
    
    dobj.unload_all()
    gc.collect()
