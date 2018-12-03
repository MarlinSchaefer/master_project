import keras

def get_model():
    model = keras.models.Sequential()
    
    model.add(keras.layers.Conv1D(64, 16, input_shape=(4096,1)))
    model.add(keras.layers.MaxPooling1D(4))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv1D(128, 16, input_shape=(4096,1)))
    model.add(keras.layers.MaxPooling1D(4))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv1D(256, 16, input_shape=(4096,1)))
    model.add(keras.layers.MaxPooling1D(4))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv1D(512, 16, input_shape=(4096,1)))
    model.add(keras.layers.MaxPooling1D(4))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128,activation='relu'))
    model.add(keras.layers.Dense(64,activation='relu'))
    model.add(keras.layers.Dense(1))
    
    return(model)
