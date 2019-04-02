import numpy as np
import keras
import load_data as ld

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
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.num_samples, self.n_channels))
        y = np.empty([self.batch_size] + list(self.label[0].shape))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i] = self.data[ID]

            # Store class
            y[i] = self.labels[ID]
        
        if not format_data == None:
            X = format_data(X)
        
        if not format_labels == None:
            y = format_labels(y)
        
        return X, y
