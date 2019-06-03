import matplotlib
matplotlib.use('Agg')
import numpy as np
import h5py
import matplotlib.pyplot as plt

def plot_false_alarm(dobj, file_path, image_path, show=True):
    with h5py.File(file_path, 'r') as FILE:
        true_vals = dobj.loaded_test_labels
        
        SNR = []
        
        #Here I take the true negatives and only record their SNR.
        for i, sample in enumerate(true_vals[1]):
            if sample[0] < sample[1]:
                #SNR.append((i, FILE['data'][i][0]))
                SNR.append((i, FILE['0'][i][0]))
        
        x_pt = sorted([pt[1] for pt in SNR])
        y_pt = []
        for pt in x_pt:
            c = 0
            for p in x_pt:
                if p > pt:
                    c += 1
            y_pt.append(c)
        
        #Total number of samples is used to determine the observation time.
        #Is this correct or should the negative samples be used?
        #0.5 = time (in seconds) that the signal is shifted around in the data
        obs_time = (len(dobj.loaded_test_labels[0])) * 0.5
        seconds_per_month = 60 * 60 * 24 * 30
        
        y_pt = [pt / obs_time * seconds_per_month for pt in y_pt]
    
    #Store to file
    store_file_path = image_path[:-4] + '.hf5'
    with h5py.File(store_file_path, 'w') as FILE:
        FILE.create_dataset('data', data=np.array([x_pt, y_pt]))
    
    plt.semilogy(x_pt, y_pt)
    plt.xlabel('SNR')
    plt.ylabel('#False alarms louder per 30 days')
    plt.title('Total number of noise samples: {}'.format(len(x_pt)))
    #plt.yscale('log')
    plt.savefig(image_path)
    if show:
        plt.show()
    else:
        plt.cla()
        plt.clf()
        plt.close()
    
    return(store_file_path)

def plot_false_alarm_prob(dobj, file_path, image_path, show=True):
    with h5py.File(file_path, 'r') as FILE:
        true_vals = dobj.loaded_test_labels
        
        probability = []
        
        #Here I take the true negatives and only record their probability value.
        #Assumes probability value is stored at the second position
        for i, sample in enumerate(true_vals[1]):
            if sample[0] < sample[1]:
                #probability.append((i, FILE['data'][i][1]))
                probability.append((i, FILE['1'][i][0]))
        
        x_pt = sorted([pt[1] for pt in probability])
        y_pt = []
        for pt in x_pt:
            c = 0
            for p in x_pt:
                if p > pt:
                    c += 1
            y_pt.append(c)
        
        #Total number of samples is used to determine the observation time.
        #Is this correct or should the negative samples be used?
        #0.5 = time (in seconds) that the signal is shifted around in the data
        obs_time = (len(dobj.loaded_test_labels[0])) * 0.5
        seconds_per_month = 60 * 60 * 24 * 30
        
        y_pt = [pt / obs_time * seconds_per_month for pt in y_pt]
    
    #Store to file
    store_file_path = image_path[:-4] + '.hf5'
    with h5py.File(store_file_path, 'w') as FILE:
        FILE.create_dataset('data', data=np.array([x_pt, y_pt]))
    
    plt.semilogy(x_pt, y_pt)
    plt.xlabel('p-value')
    plt.ylabel('#False alarms louder per 30 days')
    plt.title('Total number of noise samples: {}'.format(len(x_pt)))
    #plt.yscale('log')
    plt.savefig(image_path)
    if show:
        plt.show()
    else:
        plt.cla()
        plt.clf()
        plt.close()
    
    return(store_file_path)

#bins=(SNR_min, SNR_max, delta_SNR)
def plot_sensitivity(dobj, file_path, false_alarm_path, image_path, bins=(10, 50, 1), show=True):
    with h5py.File(file_path, 'r') as predFile:
        try:
            with h5py.File(false_alarm_path, 'r') as falseAlarmFile:
                true_vals = dobj.loaded_test_labels
                max_false_snr = -np.inf
                snr_vals = []
                for i in range(len(true_vals[1])):
                    #if true_vals[1][i][0] <= true_vals[1][i][1]:
                        #if predFile['data'][i][1] > predFile['data'][i][2]:
                            #max_false_snr = max(max_false_snr, predFile['data'][i][0])
                    #else:
                        #snr_vals.append([true_vals[0][i][0], predFile['data'][i][0]])
                    if true_vals[1][i][0] <= true_vals[1][i][1]:
                        if predFile['1'][i][0] > predFile['1'][i][1]:
                            max_false_snr = max(max_false_snr, predFile['0'][i][0])
                    else:
                        snr_vals.append([true_vals[0][i][0], predFile['0'][i][0]])
                
                print("True vals: {}".format(true_vals))
                print("snr_vals: {}".format(snr_vals))
                print("Max_false: {}".format(max_false_snr))
                act_bins = np.arange(bins[0], bins[1], bins[2])
                SNR_bins = np.zeros(len(act_bins)+1)
                norm_factor = np.zeros(len(act_bins)+1)
                bin_order = np.digitize([pt[0] for pt in snr_vals], act_bins)
                for i in range(len(snr_vals)):
                    norm_factor[bin_order[i]] += 1
                    if snr_vals[i][1] > max_false_snr:
                        SNR_bins[bin_order[i]] += 1
                
                print("SNR_bins: {}".format(SNR_bins))
                print("norm_factor: {}".format(norm_factor))
                
                y_pt = [SNR_bins[i] / norm_factor[i] if not norm_factor[i] == 0 else 0 for i in range(len(SNR_bins))]
        except IOError:
            raise ValueError('You need to create a false alarm plot first. Please use the method "plot_false_alarm" of the metrics module to do this.')
    
    store_file_path = image_path[:-4] + '.hf5'
    with h5py.File(store_file_path, 'w') as FILE:
        FILE.create_dataset('bins', data=act_bins)
        FILE.create_dataset('data', data=np.array(y_pt))
    
    plt.bar(np.arange(bins[0]-float(bins[2]) / 2, bins[1]+float(bins[2]) / 2, bins[2]), y_pt, width=bins[2])
    #plt.hist(np.arange(bins[0]-float(bins[2]) / 2, bins[1]+float(bins[2]) / 2, bins[2]), len(np.arange(bins[0]-float(bins[2]) / 2, bins[1]+float(bins[2]) / 2, bins[2])), weights=y_pt)
    plt.xlabel('SNR')
    plt.ylabel('Fraction of signals louder than highest false positive')
    plt.title('Loudest false positive SNR-value: {}'.format(max_false_snr))
    plt.savefig(image_path)
    if show:
        plt.show()
    else:
        plt.cla()
        plt.clf()
        plt.close()
    
    return(store_file_path)

def plot_sensitivity_prob(dobj, file_path, false_alarm_path, image_path, bins=(0, 1, 0.01), show=True):
    with h5py.File(file_path, 'r') as predFile:
        try:
            with h5py.File(false_alarm_path, 'r') as falseAlarmFile:
                true_vals = dobj.loaded_test_labels
                max_false_prob = -np.inf
                prob_vals = []
                for i in range(len(true_vals[1])):
                    #if true_vals[1][i][0] <= true_vals[1][i][1]:
                        #if predFile['data'][i][1] > predFile['data'][i][2]:
                            #max_false_prob = max(max_false_prob, predFile['data'][i][1])
                    #else:
                        #prob_vals.append([true_vals[1][i][0], predFile['data'][i][1]])
                    if true_vals[1][i][0] <= true_vals[1][i][1]:
                        if predFile['1'][i][0] > predFile['1'][i][1]:
                            max_false_prob = max(max_false_prob, predFile['1'][i][0])
                    else:
                        prob_vals.append([true_vals[1][i][0], predFile['1'][i][0]])
                
                print("True vals: {}".format(true_vals))
                print("prob_vals: {}".format(prob_vals))
                print("Max_false: {}".format(max_false_prob))
                act_bins = np.arange(bins[0], bins[1], bins[2])
                prob_bins = np.zeros(len(act_bins)+1)
                norm_factor = np.zeros(len(act_bins)+1)
                bin_order = np.digitize([pt[0] for pt in prob_vals], act_bins)
                for i in range(len(prob_vals)):
                    norm_factor[bin_order[i]] += 1
                    if prob_vals[i][1] > max_false_prob:
                        prob_bins[bin_order[i]] += 1
                
                print("prob_bins: {}".format(prob_bins))
                print("norm_factor: {}".format(norm_factor))
                
                y_pt = [prob_bins[i] / norm_factor[i] if not norm_factor[i] == 0 else 0 for i in range(len(prob_bins))]
        except IOError:
            raise ValueError('You need to create a false alarm plot first. Please use the method "plot_false_alarm" of the metrics module to do this.')
    
    store_file_path = image_path[:-4] + '.hf5'
    with h5py.File(store_file_path, 'w') as FILE:
        FILE.create_dataset('bins', data=act_bins)
        FILE.create_dataset('data', data=np.array(y_pt))
    
    plt.bar(np.arange(bins[0]-float(bins[2]) / 2, bins[1]+float(bins[2]) / 2, bins[2]), y_pt, width=bins[2])
    #plt.hist(np.arange(bins[0]-float(bins[2]) / 2, bins[1]+float(bins[2]) / 2, bins[2]), len(np.arange(bins[0]-float(bins[2]) / 2, bins[1]+float(bins[2]) / 2, bins[2])), weights=y_pt)
    plt.xlabel('probability')
    plt.ylabel('Fraction of signals louder than highest false positive')
    plt.title('Loudest false positive p-value: {}'.format(max_false_prob))
    plt.savefig(image_path)
    if show:
        plt.show()
    else:
        plt.cla()
        plt.clf()
        plt.close()
    
    return(store_file_path)
