import keras
import numpy as np
import h5py

def store_test_results(net, dobj, store_path):
    print("Calculating data to store.")
    res = net.predict_generator(generator, verbose=1)
    
    if type(res) == list:
        shape = [0, 0]
        shape[0] = len(res[0])
        for con in res:
            shape[1] += con.chape[-1]
        st = np.empty(shape).transpose()
        j = 0
        for con in res:
            cur = con.transpose()
            for i in len(cur):
                st[j] = cur[i]
                j += 1
        st = st.transpose()
        with h5py.File(store_path, 'w+') as FILE:
            FILE.create_dataset('data', data=st)
    else:
        with h5py.File(store_path, 'w+') as FILE:
            FILE.create_dataset('data', data=res)
    
    print("Stored data at: {}\n".format(store_path))
    
