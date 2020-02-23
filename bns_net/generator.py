import numpy as np
import keras
import load_data as ld
from pycbc.types import TimeSeries
from pycbc.filter import resample_to_delta_t
from pycbc.psd import aLIGOZeroDetHighPower
from generate_split_data import whiten_data, whiten_data_new

"""
Disclaimer:
The following code is closely guided by:
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""

class generatorFromData(keras.utils.Sequence):
    def __init__(self, data, labels, batch_size=32, shuffle=True):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.data_is_list = isinstance(self.data, list)
        self.label_is_list = isinstance(self.labels, list)
        self.shuffle = shuffle
        self._check_data()
    
    def _check_data(self):
        if self.data_is_list:
            self.n_entries = len(self.data[0])
            for dat in self.data:
                assert len(dat) == self.n_entries
        else:
            self.n_entries = len(self.data)
        
        if self.label_is_list:
            for lab in self.labels:
                assert len(lab) == self.n_entries
        else:
            assert len(self.labels) == self.n_entries
    
    def __len__(self):
        return int(np.ceil(float(self.n_entries) / self.batch_size))
    
    def __getitem__(self, index):
        start = index * self.batch_size
        if start + self.batch_size > self.n_entries:
            stop = self.n_entries - start
        else:
            stop = start + self.batch_size
       
        if self.data_is_list:
            X = [dat[start:stop] for dat in self.data]
        else:
            X = self.data[start:stop]
        
        if self.label_is_list:
            y = [lab[start:stop] for lab in self.labels]
        else:
            y = self.labels[start:stop]
        return X, y
    
    def on_epoch_end(self):
        indices = np.arange(self.n_entries, dtype=int)
        if self.data_is_list:
            self.data = [dat[indices] for dat in self.data]
        else:
            self.data = self.data[indices]
        
        if self.label_is_list:
            self.labels = [lab[indices] for lab in self.labels]
        else:
            self.labels = self.labels[indices]
        

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, labels, batch_size=32, num_samples=4096, n_channels=14, shuffle=True, format_data=None, format_label=None):
        'Initialization'
        self.data = data
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = np.arange(len(data))
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.format_data = format_data
        self.format_label=format_label
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(float(len(self.list_IDs)) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if (index+1)*self.batch_size < len(self.indexes):
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        else:
            indexes = self.indexes[index*self.batch_size:]

        # Generate data
        X, y = self.__data_generation(indexes)
        
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty([len(list_IDs_temp)] + list(self.data[0].shape))
        
        y_1 = np.empty((len(list_IDs_temp), 1))
        
        y_2 = np.empty((len(list_IDs_temp), 2))
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i] = self.data[ID]

            # Store class
            y_1[i] = self.labels[0][ID]
            y_2[i] = self.labels[1][ID]
        
        if not self.format_data == None:
            X = format_data(X)
        
        #if not format_labels == None:
            #y = format_labels(y)
        
        return X, [y_1, y_2]

class DataGeneratorMultInput(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, labels, batch_size=32, num_samples=4096, n_channels=14, shuffle=True, format_data=None, format_label=None):
        'Initialization'
        self.data = data
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = np.arange(len(data[1]))
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.format_data = format_data
        self.format_label=format_label
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(float(len(self.list_IDs)) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if (index+1)*self.batch_size < len(self.indexes):
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        else:
            indexes = self.indexes[index*self.batch_size:]

        # Generate data
        X, y = self.__data_generation(indexes)
        
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data[1]))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = [np.empty([len(list_IDs_temp)] + list(self.data[0].shape[-2:])) for j in range(len(self.data))]
        
        y_1 = np.empty((len(list_IDs_temp), 1))
        
        y_2 = np.empty((len(list_IDs_temp), 2))
        
        # Generate data
        for j in range(len(X)):
            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                X[j][i] = self.data[j][ID]

                # Store class
                y_1[i] = self.labels[0][ID]
                y_2[i] = self.labels[1][ID]
        
        if not self.format_data == None:
            X = format_data(X)
        
        #if not format_labels == None:
            #y = format_labels(y)
        
        return X, [y_1, y_2]

class generatorFromTimeSeriesAllChannels(keras.utils.Sequence):
    def __init__(self, ts, time_step=0.25, batch_size=32, dt=None):
        self.batch_size = batch_size
        if not isinstance(ts, list):
            ts = [ts]
        self.ts = []
        self.dt = []
        for t in ts:
            if isinstance(t, TimeSeries):
                self.dt.append(t.delta_t)
                self.ts.append(t)
            elif isinstance(t, type(np.array([]))):
                if dt == None:
                    msg  = 'If the provided data is not a pycbc.types.TimeSeries'
                    msg += 'a value dt must be provided.'
                    raise ValueError(msg)
                else:
                    self.dt.append(dt)
                    self.ts.append(TimeSeries(t, delta_t=dt))
            else:
                msg  = 'The provided data needs to be either a list or a '
                msg += 'single instance of either a pycbc.types.TimeSeries'
                msg += 'or a numpy.array.'
                raise ValueError(msg)
        
        for delta_t in self.dt:
            if not delta_t == self.dt[0]:
                raise ValueError('All data must have the same delta_t.')
        
        self.final_data_samples = 4096
        #The delta_t for all data
        self.dt = self.dt[0]
        #How big is the window that is shifted over the data
        #(64s + 4s for cropping when whitening)
        self.window_size_time = 68.0
        self.window_size = int(self.window_size_time / self.dt)
        #How many points are shifted each step
        self.stride = int(time_step / self.dt)
        #How many window shifts happen
        self.window_shifts = int(np.floor(float(len(self.ts[0])-self.window_size) / self.stride))
        
        self.resample_rates = [1.0 / 4096, 1.0 / 2048, 1.0 / 1024,
                               1.0 / 512, 1.0 / 256, 1.0 / 128, 1.0 / 64]
    
    def __len__(self):
        return(int(np.ceil(float(self.window_shifts) / self.batch_size)))
    
    def __getitem__(self, index):
        min_stride = index * self.batch_size
        max_stride = min_stride + self.batch_size
        if index == len(self) - 1:
            len_data = (index + 1) * self.stride * self.batch_size
            if len_data > len(self.ts[0]):
                max_stride -= int(np.floor(float(len(self.ts[0]) - len_data + self.batch_size) / self.stride))
        index_range = np.zeros((2, max_stride - min_stride), dtype=int)
        index_range[0] = np.arange(min_stride * self.stride, max_stride * self.stride, self.stride)
        index_range[1] = index_range[0] + self.window_size
        index_range = index_range.transpose()
        
        X = self._gen_slices(index_range)
        
        return(X)
    
    def _gen_slices(self, index_range):
        X = [np.zeros((len(index_range), len(self.ts) * 7, self.final_data_samples)) for i in range(2)]
        
        for detector in range(len(self.ts)):
            for in_batch, idx in enumerate(index_range):
                for i, sr in enumerate(self.resample_rates):
                    low, up = idx
                    X[0][in_batch][i * len(self.ts) + detector] = self.resample_and_whiten(self.ts[detector][low:up], sr)
                print("{}: in_batch: {}".format(detector, in_batch))
        
        return([X[0].transpose(0, 2, 1), X[1]])
    
    def resample_and_whiten(self, data, dt):
        to_resample = data.whiten(4, 4)
        if dt == self.dt:
            return(np.array(to_resample.data[-1 * self.final_data_samples:]))
        else:
            return(np.array(resample_to_delta_t(to_resample, dt).data[-1 * self.final_data_samples:]))
    
    def on_epoch_end(self):
        return

class generatorFromTimeSeriesReduced(keras.utils.Sequence):
    from generate_split_data import whiten_data
    def __init__(self, ts, time_step=0.25, batch_size=32, dt=None):
        self.batch_size = batch_size
        if not isinstance(ts, list):
            ts = [ts]
        self.ts = []
        self.dt = []
        for t in ts:
            if isinstance(t, TimeSeries):
                self.dt.append(t.delta_t)
                self.ts.append(t)
            elif isinstance(t, type(np.array([]))):
                if dt == None:
                    msg  = 'If the provided data is not a pycbc.types.TimeSeries'
                    msg += 'a value dt must be provided.'
                    raise ValueError(msg)
                else:
                    self.dt.append(dt)
                    self.ts.append(TimeSeries(t, delta_t=dt))
            else:
                msg  = 'The provided data needs to be either a list or a '
                msg += 'single instance of either a pycbc.types.TimeSeries'
                msg += 'or a numpy.array.'
                raise ValueError(msg)
        
        for delta_t in self.dt:
            if not delta_t == self.dt[0]:
                raise ValueError('All data must have the same delta_t.')
        
        self.final_data_samples = 2048
        #The delta_t for all data
        self.dt = self.dt[0]
        #How big is the window that is shifted over the data
        #(64s + 4s for cropping when whitening)
        self.window_size_time = 96.0
        self.window_size = int(self.window_size_time / self.dt)
        #How many points are shifted each step
        self.stride = int(time_step / self.dt)
        #How many window shifts happen
        self.window_shifts = int(np.floor(float(len(self.ts[0])-self.window_size) / self.stride))
        
        self.resample_dt = [1.0 / 4096, 1.0 / 2048, 1.0 / 1024,
                               1.0 / 512, 1.0 / 256, 1.0 / 128, 1.0 / 64]
        self.resample_rates = [4096, 4096, 2048, 1024, 512, 256, 128, 64]
        self.num_samples = 2048
        
        DF = 1.0 / 96
        F_LEN = int(2.0 / (DF * self.dt))
        self.psd = aLIGOZeroDetHighPower(length=F_LEN, delta_f=DF, low_freq_cutoff=20.0)
    
    def __len__(self):
        return(int(np.ceil(float(self.window_shifts) / self.batch_size))-1)
    
    def __getitem__(self, index):
        min_stride = index * self.batch_size
        max_stride = min_stride + self.batch_size
        if index == len(self) - 1:
            len_data = (index + 1) * self.stride * self.batch_size
            if len_data > len(self.ts[0]):
                max_stride -= int(np.floor(float(len(self.ts[0]) - len_data + self.batch_size) / self.stride))
        index_range = np.zeros((2, max_stride - min_stride), dtype=int)
        index_range[0] = np.arange(min_stride * self.stride, max_stride * self.stride, self.stride)
        index_range[1] = index_range[0] + self.window_size
        index_range = index_range.transpose()
        
        X = self._gen_slices(index_range)
        
        return(X)
    
    def _gen_slices(self, index_range):
        X = [np.zeros((len(index_range), len(self.ts) * 8, self.final_data_samples)) for i in range(2)]
        
        for in_batch, idx in enumerate(index_range):
            for detector in range(len(self.ts)):
                low, up = idx
                #white_full_signal = self.ts[detector][low:up].whiten(4, 4)
                white_full_signal = whiten_data(self.ts[detector][low:up], psd=self.psd)
                max_idx = len(white_full_signal)
                min_idx = max_idx - int(float(self.num_samples) / float(self.resample_rates[0]) / self.dt)
                for i, sr in enumerate(self.resample_rates):
                    X[0][in_batch][i * len(self.ts) + detector] = resample_to_delta_t(white_full_signal[min_idx:max_idx], 1.0 / sr)
                    if not i + 1 == len(self.resample_rates):
                        t_dur = float(self.num_samples) / float(self.resample_rates[i+1])
                        sample_dur = int(t_dur / self.dt)
                        max_idx = min_idx
                        min_idx -= sample_dur
                #print("{}: in_batch: {}".format(detector, in_batch))
        
        return([X[0].transpose(0, 2, 1), X[1].transpose(0, 2, 1)])
    
    def resample_and_whiten(self, data, dt):
        to_resample = data.whiten(4, 4)
        if dt == self.dt:
            return(np.array(to_resample.data[-1 * self.final_data_samples:]))
        else:
            return(np.array(resample_to_delta_t(to_resample, dt).data[-1 * self.final_data_samples:]))
    
    def on_epoch_end(self):
        return

class generatorFromTimeSeriesReducedSplit(keras.utils.Sequence):
    from generate_split_data import whiten_data_new
    def __init__(self, ts, time_step=0.25, batch_size=32, dt=None):
        self.batch_size = batch_size
        if not isinstance(ts, list):
            ts = [ts]
        self.ts = []
        self.dt = []
        for t in ts:
            if isinstance(t, TimeSeries):
                self.dt.append(t.delta_t)
                self.ts.append(t)
            elif isinstance(t, type(np.array([]))):
                if dt == None:
                    msg  = 'If the provided data is not a pycbc.types.TimeSeries'
                    msg += 'a value dt must be provided.'
                    raise ValueError(msg)
                else:
                    self.dt.append(dt)
                    self.ts.append(TimeSeries(t, delta_t=dt))
            else:
                msg  = 'The provided data needs to be either a list or a '
                msg += 'single instance of either a pycbc.types.TimeSeries'
                msg += 'or a numpy.array.'
                raise ValueError(msg)
        
        for delta_t in self.dt:
            if not delta_t == self.dt[0]:
                raise ValueError('All data must have the same delta_t.')
        
        self.final_data_samples = 2048
        #The delta_t for all data
        self.dt = self.dt[0]
        #How big is the window that is shifted over the data
        #(64s + 8s for cropping when whitening)
        self.window_size_time = 72.0
        self.window_size = int(self.window_size_time / self.dt)
        #How many points are shifted each step
        self.stride = int(time_step / self.dt)
        #How many window shifts happen
        self.window_shifts = int(np.floor(float(len(self.ts[0])-self.window_size + self.stride) / self.stride))
        
        self.resample_dt = [1.0 / 4096, 1.0 / 2048, 1.0 / 1024,
                               1.0 / 512, 1.0 / 256, 1.0 / 128, 1.0 / 64]
        self.resample_rates = [4096, 4096, 2048, 1024, 512, 256, 128, 64]
        self.num_samples = 2048
        
        #create PSD for whitening
        DF = 1.0 / 96
        F_LEN = int(2.0 / (DF * self.dt))
        self.psd = aLIGOZeroDetHighPower(length=F_LEN, delta_f=DF, low_freq_cutoff=20.0)
    
    def __len__(self):
        return(int(np.ceil(float(self.window_shifts) / self.batch_size))-1)
    
    def __getitem__(self, index):
        min_stride = index * self.batch_size
        max_stride = min_stride + self.batch_size
        if index == len(self) - 1:
            len_data = (index + 1) * self.stride * self.batch_size
            if len_data > len(self.ts[0]):
                max_stride -= int(np.floor(float(len(self.ts[0]) - len_data + self.batch_size) / self.stride))
        index_range = np.zeros((2, max_stride - min_stride), dtype=int)
        index_range[0] = np.arange(min_stride * self.stride, max_stride * self.stride, self.stride)
        index_range[1] = index_range[0] + self.window_size
        index_range = index_range.transpose()
        
        X = self._gen_slices(index_range)
        
        return(X)
    
    def _gen_slices(self, index_range):
        num_channels = 8
        num_detectors = 2
        X = [np.zeros((len(index_range), len(self.ts) * 8, self.final_data_samples)) for i in range(2)]
        X_ret = [np.zeros((len(self.ts), self.final_data_samples, len(index_range))) for i in range(num_detectors*num_channels)]
        
        for in_batch, idx in enumerate(index_range):
            for detector in range(len(self.ts)):
                low, up = idx
                #Using whiten_data works
                white_full_signal = whiten_data_new(self.ts[detector][low:up])
                #white_full_signal = self.ts[detector][low:up].whiten(4, 4)
                max_idx = len(white_full_signal)
                min_idx = max_idx - int(float(self.num_samples) / float(self.resample_rates[0]) / self.dt)
                for i, sr in enumerate(self.resample_rates):
                    X[0][in_batch][i * len(self.ts) + detector] = resample_to_delta_t(white_full_signal[min_idx:max_idx], 1.0 / sr)
                    if not i + 1 == len(self.resample_rates):
                        t_dur = float(self.num_samples) / float(self.resample_rates[i+1])
                        sample_dur = int(t_dur / self.dt)
                        max_idx = min_idx
                        min_idx -= sample_dur
                #print("{}: in_batch: {}".format(detector, in_batch))
        
        X[0] = X[0].transpose(1, 2, 0)
        X[1] = X[1].transpose(1, 2, 0)
        
        X_ret[0][0] = X[0][0]
        X_ret[0][1] = X[0][1]
        X_ret[1][0] = X[1][0]
        X_ret[1][1] = X[1][1]
        X_ret[2][0] = X[0][2]
        X_ret[2][1] = X[0][3]
        X_ret[3][0] = X[1][2]
        X_ret[3][1] = X[1][3]
        X_ret[4][0] = X[0][4]
        X_ret[4][1] = X[0][5]
        X_ret[5][0] = X[1][4]
        X_ret[5][1] = X[1][5]
        X_ret[6][0] = X[0][6]
        X_ret[6][1] = X[0][7]
        X_ret[7][0] = X[1][6]
        X_ret[7][1] = X[1][7]
        X_ret[8][0] = X[0][8]
        X_ret[8][1] = X[0][9]
        X_ret[9][0] = X[1][8]
        X_ret[9][1] = X[1][9]
        X_ret[10][0] = X[0][10]
        X_ret[10][1] = X[0][11]
        X_ret[11][0] = X[1][10]
        X_ret[11][1] = X[1][11]
        X_ret[12][0] = X[0][12]
        X_ret[12][1] = X[0][13]
        X_ret[13][0] = X[1][12]
        X_ret[13][1] = X[1][13]
        X_ret[14][0] = X[0][14]
        X_ret[14][1] = X[0][15]
        X_ret[15][0] = X[1][14]
        X_ret[15][1] = X[1][15]
        
        return([x.transpose(2, 1, 0) for x in X_ret])
    
    def resample_and_whiten(self, data, dt):
        to_resample = data.whiten(4, 4)
        if dt == self.dt:
            return(np.array(to_resample.data[-1 * self.final_data_samples:]))
        else:
            return(np.array(resample_to_delta_t(to_resample, dt).data[-1 * self.final_data_samples:]))
    
    def on_epoch_end(self):
        return

class generatorFromTimeSeriesReducedSplitRemoveLast(keras.utils.Sequence):
    from generate_split_data import whiten_data_new
    def __init__(self, ts, time_step=0.25, batch_size=32, dt=None):
        self.batch_size = batch_size
        if not isinstance(ts, list):
            ts = [ts]
        self.ts = []
        self.dt = []
        for t in ts:
            if isinstance(t, TimeSeries):
                self.dt.append(t.delta_t)
                self.ts.append(t)
            elif isinstance(t, type(np.array([]))):
                if dt == None:
                    msg  = 'If the provided data is not a pycbc.types.TimeSeries'
                    msg += 'a value dt must be provided.'
                    raise ValueError(msg)
                else:
                    self.dt.append(dt)
                    self.ts.append(TimeSeries(t, delta_t=dt))
            else:
                msg  = 'The provided data needs to be either a list or a '
                msg += 'single instance of either a pycbc.types.TimeSeries'
                msg += 'or a numpy.array.'
                raise ValueError(msg)
        
        for delta_t in self.dt:
            if not delta_t == self.dt[0]:
                raise ValueError('All data must have the same delta_t.')
        
        self.final_data_samples = 2048
        #The delta_t for all data
        self.dt = self.dt[0]
        #How big is the window that is shifted over the data
        #(64s + 8s for cropping when whitening)
        self.window_size_time = 72.0
        self.window_size = int(self.window_size_time / self.dt)
        #How many points are shifted each step
        self.stride = int(time_step / self.dt)
        #How many window shifts happen
        self.window_shifts = int(np.floor(float(len(self.ts[0])-self.window_size + self.stride) / self.stride))
        
        self.resample_dt = [1.0 / 4096, 1.0 / 2048, 1.0 / 1024,
                               1.0 / 512, 1.0 / 256, 1.0 / 128]
        self.resample_rates = [4096, 4096, 2048, 1024, 512, 256, 128]
        self.num_samples = 2048
        
        #create PSD for whitening
        DF = 1.0 / 96
        F_LEN = int(2.0 / (DF * self.dt))
        self.psd = aLIGOZeroDetHighPower(length=F_LEN, delta_f=DF, low_freq_cutoff=20.0)
    
    def __len__(self):
        return(int(np.ceil(float(self.window_shifts) / self.batch_size))-1)
    
    def __getitem__(self, index):
        min_stride = index * self.batch_size
        max_stride = min_stride + self.batch_size
        if index == len(self) - 1:
            len_data = (index + 1) * self.stride * self.batch_size
            if len_data > len(self.ts[0]):
                max_stride -= int(np.floor(float(len(self.ts[0]) - len_data + self.batch_size) / self.stride))
        index_range = np.zeros((2, max_stride - min_stride), dtype=int)
        index_range[0] = np.arange(min_stride * self.stride, max_stride * self.stride, self.stride)
        index_range[1] = index_range[0] + self.window_size
        index_range = index_range.transpose()
        
        X = self._gen_slices(index_range)
        
        return(X)
    
    def _gen_slices(self, index_range):
        num_channels = 7
        num_detectors = 2
        X = [np.zeros((len(index_range), len(self.ts) * num_channels, self.final_data_samples)) for i in range(2)]
        X_ret = [np.zeros((len(self.ts), self.final_data_samples, len(index_range))) for i in range(num_detectors*num_channels)]
        
        for in_batch, idx in enumerate(index_range):
            for detector in range(len(self.ts)):
                low, up = idx
                #Using whiten_data works
                white_full_signal = whiten_data_new(self.ts[detector][low:up])
                #white_full_signal = self.ts[detector][low:up].whiten(4, 4)
                max_idx = len(white_full_signal)
                min_idx = max_idx - int(float(self.num_samples) / float(self.resample_rates[0]) / self.dt)
                for i, sr in enumerate(self.resample_rates):
                    X[0][in_batch][i * len(self.ts) + detector] = resample_to_delta_t(white_full_signal[min_idx:max_idx], 1.0 / sr)
                    if not i + 1 == len(self.resample_rates):
                        t_dur = float(self.num_samples) / float(self.resample_rates[i+1])
                        sample_dur = int(t_dur / self.dt)
                        max_idx = min_idx
                        min_idx -= sample_dur
                #print("{}: in_batch: {}".format(detector, in_batch))
        
        X[0] = X[0].transpose(1, 2, 0)
        X[1] = X[1].transpose(1, 2, 0)
        
        X_ret[0][0] = X[0][0]
        X_ret[0][1] = X[0][1]
        X_ret[1][0] = X[1][0]
        X_ret[1][1] = X[1][1]
        X_ret[2][0] = X[0][2]
        X_ret[2][1] = X[0][3]
        X_ret[3][0] = X[1][2]
        X_ret[3][1] = X[1][3]
        X_ret[4][0] = X[0][4]
        X_ret[4][1] = X[0][5]
        X_ret[5][0] = X[1][4]
        X_ret[5][1] = X[1][5]
        X_ret[6][0] = X[0][6]
        X_ret[6][1] = X[0][7]
        X_ret[7][0] = X[1][6]
        X_ret[7][1] = X[1][7]
        X_ret[8][0] = X[0][8]
        X_ret[8][1] = X[0][9]
        X_ret[9][0] = X[1][8]
        X_ret[9][1] = X[1][9]
        X_ret[10][0] = X[0][10]
        X_ret[10][1] = X[0][11]
        X_ret[11][0] = X[1][10]
        X_ret[11][1] = X[1][11]
        X_ret[12][0] = X[0][12]
        X_ret[12][1] = X[0][13]
        X_ret[13][0] = X[1][12]
        X_ret[13][1] = X[1][13]
        
        return([x.transpose(2, 1, 0) for x in X_ret])
    
    def resample_and_whiten(self, data, dt):
        to_resample = data.whiten(4, 4)
        if dt == self.dt:
            return(np.array(to_resample.data[-1 * self.final_data_samples:]))
        else:
            return(np.array(resample_to_delta_t(to_resample, dt).data[-1 * self.final_data_samples:]))
    
    def on_epoch_end(self):
        return

class experimentalGeneratorSmall(keras.utils.Sequence):
    #from generate_split_data import whiten_data, resample_data
    def __init__(self, ts, time_step=0.25, batch_size=32, dt=None):
        #from generate_split_data import whiten_data, resample_data
        self.batch_size = batch_size
        if not isinstance(ts, list):
            ts = [ts]
        self.ts = []
        self.dt = []
        for t in ts:
            if isinstance(t, TimeSeries):
                self.dt.append(t.delta_t)
                self.ts.append(t)
            elif isinstance(t, type(np.array([]))):
                if dt == None:
                    msg  = 'If the provided data is not a pycbc.types.TimeSeries'
                    msg += 'a value dt must be provided.'
                    raise ValueError(msg)
                else:
                    self.dt.append(dt)
                    self.ts.append(TimeSeries(t, delta_t=dt))
            else:
                msg  = 'The provided data needs to be either a list or a '
                msg += 'single instance of either a pycbc.types.TimeSeries'
                msg += 'or a numpy.array.'
                raise ValueError(msg)
        
        for delta_t in self.dt:
            if not delta_t == self.dt[0]:
                raise ValueError('All data must have the same delta_t.')
        
        self.final_data_samples = 4096
        #The delta_t for all data
        self.dt = self.dt[0]
        #How big is the window that is shifted over the data
        #(64s + 4s for cropping when whitening)
        self.window_size_time = 96.0
        self.window_size = int(self.window_size_time / self.dt)
        #How many points are shifted each step
        self.stride = int(time_step / self.dt)
        #How many window shifts happen
        self.window_shifts = int(np.floor(float(len(self.ts[0])-self.window_size) / self.stride))
        
        self.resample_dt = [1.0 / 4096, 1.0 / 2048, 1.0 / 1024,
                               1.0 / 512, 1.0 / 256, 1.0 / 128, 1.0 / 64]
        self.resample_rates = [4096, 4096, 2048, 1024, 512, 256, 128, 64]
        self.num_samples = 4096
        
        DF = 1.0 / 96
        F_LEN = int(2.0 / (DF * self.dt))
        self.psd = aLIGOZeroDetHighPower(length=F_LEN, delta_f=DF, low_freq_cutoff=20.0)
    
    def __len__(self):
        return(int(np.ceil(float(self.window_shifts) / self.batch_size))-1)
    
    def __getitem__(self, index):
        min_stride = index * self.batch_size
        max_stride = min_stride + self.batch_size
        if index == len(self) - 1:
            len_data = (index + 1) * self.stride * self.batch_size
            if len_data > len(self.ts[0]):
                max_stride -= int(np.floor(float(len(self.ts[0]) - len_data + self.batch_size) / self.stride))
        index_range = np.zeros((2, max_stride - min_stride), dtype=int)
        index_range[0] = np.arange(min_stride * self.stride, max_stride * self.stride, self.stride)
        index_range[1] = index_range[0] + self.window_size
        index_range = index_range.transpose()
        
        X = self._gen_slices(index_range)
        
        return(X)
    
    def _gen_slices(self, index_range):
        from generate_split_data import whiten_data, resample_data
        X = [np.zeros((len(index_range), len(self.ts) * 3, self.final_data_samples)) for i in range(2)]
        
        for in_batch, idx in enumerate(index_range):
            low, up = idx
            white_list = []
            for detector in range(len(self.ts)):
                white_list.append(self.ts[detector][low:up])
            white_list = whiten_data(white_list, psd=self.psd)
            resampled_data = resample_data(white_list, self.resample_rates).transpose()
            
            X[0][in_batch][0] = resampled_data[1]
            X[0][in_batch][1] = resampled_data[8]
            X[0][in_batch][2] = resampled_data[3]
            X[0][in_batch][3] = resampled_data[10]
            X[0][in_batch][4] = resampled_data[5]
            X[0][in_batch][5] = resampled_data[12]
        
        return([X[0].transpose(0, 2, 1), X[1].transpose(0, 2, 1)])
    
    def resample_and_whiten(self, data, dt):
        to_resample = data.whiten(4, 4)
        if dt == self.dt:
            return(np.array(to_resample.data[-1 * self.final_data_samples:]))
        else:
            return(np.array(resample_to_delta_t(to_resample, dt).data[-1 * self.final_data_samples:]))
    
    def on_epoch_end(self):
        return

class experimentalGenerator(keras.utils.Sequence):
    #from generate_split_data import whiten_data, resample_data
    def __init__(self, ts, time_step=0.25, batch_size=32, dt=None):
        #from generate_split_data import whiten_data, resample_data
        self.batch_size = batch_size
        if not isinstance(ts, list):
            ts = [ts]
        self.ts = []
        self.dt = []
        for t in ts:
            if isinstance(t, TimeSeries):
                self.dt.append(t.delta_t)
                self.ts.append(t)
            elif isinstance(t, type(np.array([]))):
                if dt == None:
                    msg  = 'If the provided data is not a pycbc.types.TimeSeries'
                    msg += 'a value dt must be provided.'
                    raise ValueError(msg)
                else:
                    self.dt.append(dt)
                    self.ts.append(TimeSeries(t, delta_t=dt))
            else:
                msg  = 'The provided data needs to be either a list or a '
                msg += 'single instance of either a pycbc.types.TimeSeries'
                msg += 'or a numpy.array.'
                raise ValueError(msg)
        
        for delta_t in self.dt:
            if not delta_t == self.dt[0]:
                raise ValueError('All data must have the same delta_t.')
        
        self.final_data_samples = 2048
        #The delta_t for all data
        self.dt = self.dt[0]
        #How big is the window that is shifted over the data
        #(64s + 4s for cropping when whitening)
        self.window_size_time = 96.0
        self.window_size = int(self.window_size_time / self.dt)
        #How many points are shifted each step
        self.stride = int(time_step / self.dt)
        #How many window shifts happen
        self.window_shifts = int(np.floor(float(len(self.ts[0])-self.window_size) / self.stride))
        
        self.resample_dt = [1.0 / 4096, 1.0 / 2048, 1.0 / 1024,
                               1.0 / 512, 1.0 / 256, 1.0 / 128, 1.0 / 64]
        self.resample_rates = [4096, 4096, 2048, 1024, 512, 256, 128, 64]
        self.num_samples = 2048
        
        DF = 1.0 / 96
        F_LEN = int(2.0 / (DF * self.dt))
        self.psd = aLIGOZeroDetHighPower(length=F_LEN, delta_f=DF, low_freq_cutoff=20.0)
    
    def __len__(self):
        return(int(np.ceil(float(self.window_shifts) / self.batch_size))-1)
    
    def __getitem__(self, index):
        min_stride = index * self.batch_size
        max_stride = min_stride + self.batch_size
        if index == len(self) - 1:
            len_data = (index + 1) * self.stride * self.batch_size
            if len_data > len(self.ts[0]):
                max_stride -= int(np.floor(float(len(self.ts[0]) - len_data + self.batch_size) / self.stride))
        index_range = np.zeros((2, max_stride - min_stride), dtype=int)
        index_range[0] = np.arange(min_stride * self.stride, max_stride * self.stride, self.stride)
        index_range[1] = index_range[0] + self.window_size
        index_range = index_range.transpose()
        
        X = self._gen_slices(index_range)
        
        return(X)
    
    def _gen_slices(self, index_range):
        from generate_split_data import whiten_data, resample_data
        X = [np.zeros((len(index_range), len(self.ts) * 8, self.final_data_samples)) for i in range(2)]
        
        for in_batch, idx in enumerate(index_range):
            low, up = idx
            white_list = []
            for detector in range(len(self.ts)):
                white_list.append(self.ts[detector][low:up])
            white_list = whiten_data(white_list, psd=self.psd)
            resampled_data = resample_data(white_list, self.resample_rates).transpose()
            
            X[0][in_batch][0] = resampled_data[0][2048:]
            X[0][in_batch][1] = resampled_data[7][2048:]
            X[0][in_batch][2] = resampled_data[0][:2048]
            X[0][in_batch][3] = resampled_data[7][:2048]
            X[0][in_batch][4] = resampled_data[1][:2048]
            X[0][in_batch][5] = resampled_data[8][:2048]
            X[0][in_batch][6] = resampled_data[2][:2048]
            X[0][in_batch][7] = resampled_data[9][:2048]
            X[0][in_batch][8] = resampled_data[3][:2048]
            X[0][in_batch][9] = resampled_data[10][:2048]
            X[0][in_batch][10] = resampled_data[4][:2048]
            X[0][in_batch][11] = resampled_data[11][:2048]
            X[0][in_batch][12] = resampled_data[5][:2048]
            X[0][in_batch][13] = resampled_data[12][:2048]
            X[0][in_batch][14] = resampled_data[6][:2048]
            X[0][in_batch][15] = resampled_data[13][:2048]
        
        return([X[0].transpose(0, 2, 1), X[1].transpose(0, 2, 1)])
    
    def resample_and_whiten(self, data, dt):
        to_resample = data.whiten(4, 4)
        if dt == self.dt:
            return(np.array(to_resample.data[-1 * self.final_data_samples:]))
        else:
            return(np.array(resample_to_delta_t(to_resample, dt).data[-1 * self.final_data_samples:]))
    
    def on_epoch_end(self):
        return
