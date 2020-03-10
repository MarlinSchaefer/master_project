import file_generators as fg
import numpy as np
import h5py
import os
from aux_functions import get_store_path

class H5pyHandeler(fg.FileHandeler):
    def __init__(self, file_path, ref_index=(0, 0)):
        super(H5pyHandeler, self).__init__(file_path)
        self.ref_index = ref_index
    
    def __contains__(self, idx):
        sig_idx, noi_idx = idx
        sig_idx += self.ref_index[0]
        noi_idx += self.ref_index[1]
        if not sig_idx == -1 + self.ref_index[0]:
            if sig_idx >= len(self.file['training/signals']):
                return False
        if noi_idx >= len(self.file['training/noise']):
            return False
        return True
    
    def __getitem__(self, idx):
        sig_idx, noi_idx = idx
        sig_idx += self.ref_index[0]
        noi_idx += self.ref_index[1]
        noise = self.file['training/noise'][noi_idx][:].transpose()
        ret_noise = []
        tmp = np.zeros((2, 2048))
        for i in range(7):
            if i == 0:
                tmp[0] = noise[0][2048:]
                tmp[1] = noise[7][2048:]
            else:
                idx = i - 1
                tmp[0] = noise[idx][:2048]
                tmp[1] = noise[idx+7][:2048]
            ret_noise.append(tmp.transpose())
        
        if sig_idx == -1 + self.ref_index[0]:
            ret_signal = [np.zeros(ret_noise[0].shape) for _ in ret_noise]
            y1 = np.array([4.])
            y2 = np.array([0., 1.])
        else:
            y1 = np.array([self.file['training/signal_labels'][sig_idx][()]])
            y2 = np.array([1., 0.])
            signal = self.file['training/signals'][sig_idx][:].transpose()
            ret_signal = []
            for i in range(7):
                if i == 0:
                    tmp[0] = signal[0][2048:]
                    tmp[1] = signal[7][2048:]
                else:
                    idx = i - 1
                    tmp[0] = signal[idx][:2048]
                    tmp[1] = signal[idx+7][:2048]
                ret_signal.append(tmp.transpose())
        
        inp = []
        for i in range(7):
            inp.append(ret_signal[i])
            inp.append(ret_noise[i])
        
        out = [y1, y2] + [y1 for _ in range(6)] + [sig for sig in ret_signal]
        return inp, out
    
    def open(self, mode='r'):
        self.file = h5py.File(self.file_path, mode)
    
    def close(self):
        self.file.close()

class H5pyMultiHandeler(fg.MultiFileHandeler):
    def __init__(self, file_handelers=None, mode='r'):
        super(H5pyMultiHandeler, self).__init__(file_handelers=file_handelers, mode=mode)
        self.input_shape = [(2048, 2) for _ in range(7)]
        self.output_shape = [[1], [2]] + [[1] for _ in range(6)] + [[2048, 2] for _ in range(7)]

def main():
    file_handeler = H5pyHandeler(os.path.join(get_store_path(), 'final_network_retrain', 'templates_to_download.hf5'))
    multi_handeler = H5pyMultiHandeler()
    multi_handeler.add_file_handeler(file_handeler)
    with multi_handeler as mh:
        generator = fg.FileGenerator(mh, [(0, 0), (-1, 0), (-1, 1), (2, 3)], batch_size=2, shuffle=True)
        print(generator[0])
    return

if __name__ == "__main__":
    main()
