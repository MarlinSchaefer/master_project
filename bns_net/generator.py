import numpy as np
import keras
import load_data as ld
from pycbc.types import TimeSeries
from pycbc.filter import resample_to_delta_t

"""
Disclaimer:
The following code is closely guided by:
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""


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
