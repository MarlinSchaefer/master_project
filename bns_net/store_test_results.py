import keras
import numpy as np
import h5py
import generator as g

def store_test_results(net, dobj, store_path, batch_size=32):
    print("Calculating data to store.")
    res = net.predict_generator(g.DataGeneratorMultInput(dobj.loaded_test_data, dobj.loaded_test_labels, batch_size=batch_size), verbose=1)
    
    if type(res) == list:
        shape = [0, 0]
        shape[0] = len(res[0])
        for con in res:
            shape[1] += con.shape[-1]
        st = np.empty(shape).transpose()
        j = 0
        for con in res:
            cur = con.transpose()
            for i in range(len(cur)):
                st[j] = cur[i]
                j += 1
        st = st.transpose()
        with h5py.File(store_path, 'w') as FILE:
            FILE.create_dataset('data', data=st)
    else:
        with h5py.File(store_path, 'w') as FILE:
            FILE.create_dataset('data', data=res)
    
    print("Stored data at: {}\n".format(store_path))
    
