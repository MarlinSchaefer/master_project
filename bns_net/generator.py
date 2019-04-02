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
        return int(np.floor(float(len(self.list_IDs)) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        print("In get item")
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        print("After setting index list")
        
        print("After temp index list")
        
        print("Now trying to call __data_generation")

        # Generate data
        X, y = self.__data_generation(indexes)
        
        print("After __data_generation, pre returning")
        
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        print("In __data_generation")
        X = np.empty((self.batch_size, self.num_samples, self.n_channels))
        print("Allocated X")
        y = np.empty([self.batch_size] + list(self.labels[0].shape))
        
        print("Allocated y, before loop, len(indexes) = {}".format(len(list_IDs_temp)))
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            print("In loop | i: {}; ID: {}".format(i,ID))
            
            # Store sample
            X[i] = self.data[ID]

            # Store class
            y[i] = self.labels[ID]
        
        print("After loop")
        
        if not format_data == None:
            X = format_data(X)
        
        if not format_labels == None:
            y = format_labels(y)
        
        print("Pre returning from __data_generation")
        
        return X, y
