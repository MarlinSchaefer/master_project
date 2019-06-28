import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import keras
import h5py
import imp
import os
from aux_functions import get_store_path
from progress_bar import progress_tracker
from generator import DataGeneratorMultInput

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
    plt.title('%s: Variance: %.2f' % (net_name.replace('_', '\_'), var))
    plt.legend()
    plt.savefig(path)
    if show:
        plt.show()
    
def _do_plot(net_name, x_pt_1, x_pt_2, y_pt, path, show=False, save_file=True):
    var_arr_1 = y_pt - x_pt_1
    var_1 = np.var(var_arr_1)
    
    var_arr_2 = y_pt - x_pt_2
    var_2 = np.var(var_arr_2)
    
    line = np.linspace(0.9*min([min(x_pt_1), min(x_pt_2), min(y_pt)]),1.1*max([max(x_pt_1), max(x_pt_2), max(y_pt)]),100)
    
    dpi = 96
    plt.figure(figsize=(1920.0/dpi, 1440.0/dpi), dpi=dpi)
    plt.rcParams.update({'font.size': 22, 'text.usetex': 'true'})
    
    plt.subplot(211)
    plt.scatter(x_pt_1, y_pt, label='Data points', marker=',', s=1)
    plt.plot(line,line,color='red', label='Ideal case')
    plt.xlabel('True SNR')
    plt.ylabel('Recovered SNR')
    plt.title('%s: Variance against true SNR: %.2f' % (net_name.replace('_', '\_'), var_1))
    plt.legend()
    plt.grid()
    plt.subplots_adjust(hspace=0.7)
    
    plt.subplot(212)
    plt.scatter(x_pt_2, y_pt, label='Data points', marker=',', s=1)
    plt.plot(line,line,color='red', label='Ideal case')
    plt.xlabel('Calculated SNR')
    plt.ylabel('Recovered SNR')
    plt.title('%s: Variance against calculated SNR: %.2f' % (net_name.replace('_', '\_'), var_2))
    plt.legend()
    plt.grid()
    
    plt.savefig(path)
    
    #Save data to file
    if save_file:
        file_path = os.path.splitext(path)[0] + '.hf5'
        with h5py.File(file_path, 'w') as save_data:
            save_data.create_dataset('x1', data=np.array(x_pt_1))
            save_data.create_dataset('x2', data=np.array(x_pt_2))
            save_data.create_dataset('y', data=np.array(y_pt))
    
    if show:
        plt.show()
    else:
        plt.cla()
        plt.clf()
        plt.close()

def plot_true_and_calc(net, data, labels, calc, path, show=False, net_name='N/A', save_file=True):
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
    
    _do_plot(net_name, x_pt_1, x_pt_2, y_pt, path, show=show, save_file=save_file)

def plot_true_and_calc_partial(net, data_path, path, net_path, batch_size=32, show=False, net_name='N/A', save_file=True):
    d_form = imp.load_source('d_form', net_path)
    with h5py.File(data_path, 'r') as data:
        te_d = data['testing']['test_data']
        te_l = data['testing']['test_labels']
        te_c = data['testing']['test_snr_calculated']
        
        x_pt_1 = np.zeros(len(te_d))
        x_pt_2 = np.zeros(len(te_d))
        y_pt = np.zeros(len(te_d))
        
        steps = int(np.ceil(float(len(te_d)) / batch_size)) - 1
        
        bar = progress_tracker(steps, name='Creating plot')
        
        for i in range(steps):
            lower = i * batch_size
            upper = (i+1) * batch_size
            
            if upper > len(te_d):
                upper = len(te_d)
            
            for j in range(lower, upper):
                x_pt_1[j] = te_l[j][0]
                x_pt_2[j] = te_c[j]
                cache = net.predict(d_form.format_data_segment(np.array([te_d[j]])))
                
                if type(cache) == list:
                    y_pt[j] = cache[0][0]
                else:
                    y_pt[j] = cache[0]
            
            bar.iterate()
    
    _do_plot(net_name, x_pt_1, x_pt_2, y_pt, path, show=show, save_file=save_file)

def plot_true_and_calc_from_file(file_path, dobj, image_path, show=False, net_name='N/A', save_file=True):
    with h5py.File(file_path, 'r') as ResFile:
        #y_pt = ResFile['data'][:].transpose()[0]
        y_pt = ResFile['0'][:].transpose()[0]
    
    x_pt_1 = dobj.loaded_test_labels
    
    if type(x_pt_1) == list:
        x_pt_1 = x_pt_1[0]
    
    x_pt_1 = x_pt_1.flatten()
    
    x_pt_2 = dobj.loaded_test_snr
    
    if type(x_pt_2) == list:
        x_pt_2 = np.array(x_pt_2)
    
    x_pt_2 = x_pt_2.flatten()
    
    _do_plot(net_name, x_pt_1, x_pt_2, y_pt, image_path, show=show, save_file=save_file)

def plot_true_from_pred_file(file_path, img_path, show=False, net_name='N/A', save_file=True):
    with h5py.File(file_path, 'r') as ResFile:
        y_pt = ResFile['0'][:].transpose()[0]
        x_pt_1 = ResFile['labels']['0'][:]
    
    x_pt_1 = x_pt_1.flatten()
    x_pt_2 = np.zeros(len(y_pt))
    
    _do_plot(net_name, x_pt_1, x_pt_2, y_pt, img_path, show=show, save_file=save_file)
