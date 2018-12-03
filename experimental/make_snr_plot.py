import matplotlib.pyplot as plt
import numpy as np
import keras

def plot(net, data, labels, path, show=False, net_name='N/A'):
    x_pt = labels.reshape(len(labels))
    y_pt = np.zeros(len(data))
    
    for i, pt in enumerate(data):
        y_pt[i] = net.predict(np.array([pt]))
    
    var_arr = y_pt - x_pt
    var = np.var(var_arr)
    
    line = np.linspace(0.9*min([min(x_pt),min(y_pt)]),1.1*max([max(x_pt),max(y_pt)]),10000)
    
    plt.scatter(x_pt,y_pt,label='Data points')
    plt.plot(line,line,color='red',label='Ideal case')
    plt.xlabel('True SNR')
    plt.ylabel('Recovered SNR')
    plt.title('%s: Variance: %.2f' % (net_name, var))
    plt.legend()
    plt.savefig(path)
    if show:
        plt.show()
    
