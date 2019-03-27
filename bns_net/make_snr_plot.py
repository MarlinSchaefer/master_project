import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import keras
import h5py

def plot(net, data, labels, path, show=False, net_name='N/A'):
    x_pt = labels.reshape(len(labels))
    y_pt = np.zeros(len(data))
    
    print("Path: %s" % path)
    
    for i, pt in enumerate(data):
        y_pt[i] = net.predict(np.array([pt]))
    
    var_arr = y_pt - x_pt
    var = np.var(var_arr)
    
    line = np.linspace(0.9*min([min(x_pt),min(y_pt)]),1.1*max([max(x_pt),max(y_pt)]),100)
    
    plt.scatter(x_pt,y_pt,label='Data points')
    plt.plot(line,line,color='red',label='Ideal case')
    plt.xlabel('True SNR')
    plt.ylabel('Recovered SNR')
    plt.title('%s: Variance: %.2f' % (net_name, var))
    plt.legend()
    plt.savefig(path)
    if show:
        plt.show()
    
def _do_plot(net_name, x_pt_1, x_pt_2, y_pt, path, show=False):
    var_arr_1 = y_pt - x_pt_1
    var_1 = np.var(var_arr_1)
    
    var_arr_2 = y_pt - x_pt_2
    var_2 = np.var(var_arr_2)
    
    line = np.linspace(0.9*min([min(x_pt_1), min(x_pt_2), min(y_pt)]),1.1*max([max(x_pt_1), max(x_pt_2), max(y_pt)]),100)
    
    plt.subplot(211)
    plt.scatter(x_pt_1, y_pt, label='Data points')
    plt.plot(line,line,color='red', label='Ideal case')
    plt.xlabel('True SNR')
    plt.ylabel('Recovered SNR')
    plt.title('%s: Variance against true SNR: %.2f' % (net_name, var_1))
    plt.legend()
    plt.subplots_adjust(hspace=0.7)
    
    plt.subplot(212)
    plt.scatter(x_pt_2, y_pt, label='Data points')
    plt.plot(line,line,color='red', label='Ideal case')
    plt.xlabel('Calculated SNR')
    plt.ylabel('Recovered SNR')
    plt.title('%s: Variance against calculated SNR: %.2f' % (net_name, var_2))
    plt.legend()
    
    plt.savefig(path)
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_true_and_calc(net, data, labels, calc, path, show=False, net_name='N/A'):
    #x_pt_1 = labels.reshape(len(labels))
    x_pt_1 = np.array([pt[0] for pt in labels])
    x_pt_2 = calc
    
    y_pt = net.predict(data)
    #y_pt = y_pt.reshape(len(y_pt))
    
    #Check for multiple outputs. If the network has more than
    #one, the SNR has to be the first output.
    if type(y_pt) == list:
        y_pt = np.array([pt[0] for pt in y_pt[0]])
    else:
        y_pt = np.array([pt[0] for pt in y_pt])
    print("x_pt_1: {}".format(x_pt_1))
    print("x_pt_2: {}".format(x_pt_2))
    print("y_pt: {}".format(y_pt))
    
    _do_plot(net_name, x_pt_1, x_pt_2, y_pt, path, show=show)

def plot_true_and_calc_partial(net, data_path, path, batch_size=32, show=False, net_name='N/A'):
    with h5py.File(data_path, 'r') as data:
        te_d = data['testing']['test_data']
        te_l = data['testing']['test_labels']
        te_c = data['testing']['test_snr_calculated']
        
        x_pt_1 = np.zeros(len(te_d))
        x_pt_2 = np.zeros(len(te_d))
        y_pt = np.zeros(len(te_d))
        
        steps = int(np.ceil(float(len(te_d)) / batch_size)) - 1
        
        for i in range(steps):
            lower = i * batch_size
            upper = (i+1) * batch_size
            
            if upper > len(te_d):
                upper = len(te_d)
            
            for j in range(lower, upper):
                x_pt_1[j] = te_l[j][0]
                x_pt_2[j] = te_c[j]
                cache = net.predict(np.array([te_d[j]]))
                
                if type(cache) == list:
                    y_pt[j] = cache[0][0]
                else:
                    y_pt[j] = cache[0]
    
    _do_plot(net_name, x_pt_1, x_pt_2, y_pt, path, show=show)
