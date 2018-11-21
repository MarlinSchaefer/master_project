import keras
import matplotlib.pyplot as plt
import numpy as np
from load_data import load_data_Conv1D

(train_data, train_labels), (test_data, test_labels) = load_data_Conv1D()

model = keras.models.load_model('classify_snr_4.hf5')

res = model.predict(np.array([test_data[0]]))

x_points = test_labels.reshape(len(test_labels))
y_points = np.zeros(len(test_labels))

for i, data in enumerate(test_data):
    if i % 100 == 0:
        print(i)
    y_points[i] = model.predict(np.array([data]))

x_lin = np.linspace(7,12,1000)

plt.scatter(x_points, y_points, label='Data points')
plt.plot(x_lin, x_lin, color='red', label='Ideal case')
plt.xlabel('True SNR')
plt.ylabel('Derived SNR')
plt.legend()
plt.show()
