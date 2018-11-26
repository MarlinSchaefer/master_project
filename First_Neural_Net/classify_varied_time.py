import keras
import numpy as np
from load_data import load_data_Conv1D

(train_data, train_labels), (test_data, test_labels) = load_data_Conv1D('varied_time_templates_new')

model = keras.models.Sequential()

model.add(keras.layers.Conv1D(64, 16, input_shape=(len(train_data[0]),1)))
model.add(keras.layers.MaxPooling1D(4))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv1D(128, 16, input_shape=(len(train_data[0]),1)))
model.add(keras.layers.MaxPooling1D(4))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv1D(256, 16, input_shape=(len(train_data[0]),1)))
model.add(keras.layers.MaxPooling1D(4))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv1D(512, 16, input_shape=(len(train_data[0]),1)))
model.add(keras.layers.MaxPooling1D(4))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dense(64,activation='relu'))
model.add(keras.layers.Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape'])

model.fit(train_data, train_labels, epochs=10)

model.save("classify_varied_time.hf5")

print(model.evaluate(test_data, test_labels))
