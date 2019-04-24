import matplotlib
matplotlib.use('Agg')
import numpy as np
import h5py
import matplotlib.pyplot as plt

def plot_false_alarm(dobj, file_path, image_path, show=True):
    with h5py.File(file_path, 'r') as FILE:
        true_vals = dobj.loaded_test_labels
        
        SNR = []
        
        for i, sample in true_vals[1]:
            if sample[0] < sample[1]:
                SNR.append((i, FILE['data'][i][0]))
        
        x_pt = [pt[1] for pt in SNR]
        y_pt = []
        for pt in x_pt:
            c = 0
            for p in x_pt:
                if p > pt:
                    c += 1
            y_pt.append(c)
        
    plt.plot(x_pt, y_pt)
    plt.xlabel('SNR')
    plt.ylabel('#False alarms louder')
    plt.savefig(image_path)
    if show:
        plt.show()
    else:
        plt.cla()
        plt.clf()
        plt.close()
