import keras
import matplotlib.pyplot as plt
import numpy as np
from load_data import load_data_Conv1D

NAME_DIC = {'classify_snr_4': 'template_bank_full', 'classify_small_snr': 'small_snr_templates', 'classify_varied_time': 'varied_time_templates_new', 'classify_varied_phase': 'template_bank_phase'}

def plot(name):
    (train_data, train_labels), (test_data, test_labels) = load_data_Conv1D(name=NAME_DIC[name])
    
    model = keras.models.load_model(name + '.hf5')
    
    res = model.predict(np.array([test_data[0]]))
    
    x_points = test_labels.reshape(len(test_labels))
    y_points = np.zeros(len(test_labels))
    
    for i, data in enumerate(test_data):
        if i % 100 == 0:
            print(i)
        y_points[i] = model.predict(np.array([data]))
    
    var_arr = y_points - x_points
    var = np.var(var_arr)
    
    x_lin = np.linspace(0,15,1000)
    
    plt.scatter(x_points, y_points, label='Data points')
    plt.plot(x_lin, x_lin, color='red', label='Ideal case')
    plt.xlabel('True SNR')
    plt.ylabel('Derived SNR')
    plt.title('Variance of the data: %.2f' % var)
    plt.legend()
    plt.savefig(name + '_snr')
    plt.show()
