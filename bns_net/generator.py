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
        self.format_data = format_data
        self.format_label=format_label
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(float(len(self.list_IDs)) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

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
        X = np.empty([self.batch_size] + list(self.data[0].shape))
        
        y_1 = np.empty((self.batch_size, 1))
        
        y_2 = np.empty((self.batch_size, 2))
        
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
        return int(np.floor(float(len(self.list_IDs)) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)
        
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        print("Max index: {}".format(len(self.data[1])))
        print("Data shape: {}".format(self.data.shape))
        self.indexes = np.arange(len(self.data[1]))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = [np.empty([self.batch_size] + list(self.data[0].shape[-2:])) for j in range(len(self.data))]
        
        print("Len X: {}".format(len(X)))
        
        y_1 = np.empty((self.batch_size, 1))
        
        y_2 = np.empty((self.batch_size, 2))
        
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

