import keras.backend as K
import keras
from keras.callbacks import  Callback
import numpy as np
import os
import matplotlib.pyplot as plt
import json
from progress_bar import progress_tracker

class SensitivityTracker(Callback):
    def __init__(self, generator, dir_path, interval=1, bins=(8,15,1),
                 plot_by_interval=True, file_name='sensitivity_history',
                 plot_name='sensitivity_history_plot', verbose=1):
        super(SensitivityTracker, self).__init__()
        self.interval = interval
        self.generator = generator
        
        self.average_sensitivity_snr_history = []
        self.average_sensitivity_prob_history = []
        self.peak_sensitivity_snr_history = []
        self.peak_sensitivity_prob_history = []
        self.total_sensitivity_snr_history = []
        self.total_sensitivity_prob_history = []
        self.loudest_sample_snr = []
        self.loudest_sample_prob = []
        
        self.bins = bins
        self.plot_by_interval = plot_by_interval
        self.dir_path = dir_path
        self.file_name = file_name
        self.plot_name = plot_name
        self.verbose = bool(verbose)
    
    def file_writer(self):
        data = {}
        data['average_snr_history'] = self.average_sensitivity_snr_history
        data['peak_snr_history'] = self.peak_sensitivity_snr_history
        data['average_prob_history'] = self.average_sensitivity_prob_history
        data['peak_prob_history'] = self.peak_sensitivity_prob_history
        data['complete_snr_history'] = self.total_sensitivity_snr_history
        data['complete_prob_history'] = self.total_sensitivity_prob_history
        data['loudest_sample_snr_history'] = self.loudest_sample_snr
        data['loudest_sample_prob_history'] = self.loudest_sample_prob
        data['snr_bins'] = list(np.arange(self.bins[0], self.bins[1], self.bins[2]) + 0.5)
        data['epochs'] = list(np.arange(self.interval, (len(self.loudest_sample_snr) + 1) * self.interval, self.interval))
        
        with open(os.path.join(self.dir_path, self.file_name + '.json'), 'w') as FILE:
            json.dump(data, FILE, indent=4)
        return
    
    def _split_snr_p_val(self, data):
        y_snr = []
        y_prob = []
        for batch in data:
            for sample in batch[0]:
                y_snr.append(sample[0])
            for sample in batch[1]:
                y_prob.append(sample[0])
        return y_snr, y_prob
    
    def _bin_data(self, true_prob, y_true, y_pred):
        bins = np.arange(self.bins[0], self.bins[1], self.bins[2])
        loud_false = -np.inf
        signal_true = []
        signal_pred = []
        for i, true_bool in enumerate(true_prob):
            if not bool(true_bool):
                if y_pred[i] > loud_false:
                    loud_false = y_pred[i]
            else:
                signal_true.append(y_true[i])
                signal_pred.append(y_pred[i])
        
        signal_indices = np.digitize(signal_true, bins)
        
        bin_values = np.zeros(len(bins), dtype=np.float64)
        norm_values = np.zeros(len(bins), dtype=np.float64)
        
        for i, idx in enumerate(signal_indices):
            if idx == len(bins):
                idx = len(bins) - 1 
            if signal_pred[i] > loud_false:
                bin_values[idx] += 1
            norm_values[idx] += 1
        
        for i, norm in enumerate(norm_values):
            if norm == 0.:
                bin_values[i] = 0
            else:
                bin_values[i] = bin_values[i] / norm
        
        return bin_values, loud_false
    
    def calculate_sensitivity(self):
        model = self.model
        y_true = []
        y_pred = []
        if self.verbose:
            bar = progress_tracker(len(self.generator), name='Calculating predictions')
        for i in range(len(self.generator)):
            x, y = self.generator.__getitem__(i)
            y_p = model.predict(x)
            y_true.append(y)
            y_pred.append(y_p)
            if self.verbose:
                bar.iterate()
        
        y_true_snr, y_true_prob = self._split_snr_p_val(y_true)
        y_pred_snr, y_pred_prob = self._split_snr_p_val(y_pred)
        
        snr_bins = np.arange(self.bins[0], self.bins[1], self.bins[2])
        
        snr_bins, snr_loud = self._bin_data(y_true_prob, y_true_snr, y_pred_snr)
        prob_bins, prob_loud = self._bin_data(y_true_prob, y_true_snr, y_pred_prob)
        
        return list(snr_bins), float(snr_loud), list(prob_bins), float(prob_loud)
    
    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.interval == 0:
            snr_tot, snr_loud, prob_tot, prob_loud = self.calculate_sensitivity()
            
            self.loudest_sample_snr.append(snr_loud)
            self.loudest_sample_prob.append(prob_loud)
            
            self.total_sensitivity_snr_history.append(snr_tot)
            self.total_sensitivity_prob_history.append(prob_tot)
            
            self.peak_sensitivity_snr_history.append(np.max(snr_tot))
            self.peak_sensitivity_prob_history.append(np.max(prob_tot))
            
            self.average_sensitivity_snr_history.append(np.sum(snr_tot) / len(snr_tot))
            self.average_sensitivity_prob_history.append(np.sum(prob_tot) / len(prob_tot))
            
            self.file_writer()
            if self.plot_by_interval:
                self.plot_history()
            print("Sensitivity SNR   | Peak: {}, Average: {}".format(self.peak_sensitivity_snr_history[-1], self.average_sensitivity_snr_history[-1]))
            print("Sensitivity p-val | Peak: {}, Average: {}".format(self.peak_sensitivity_prob_history[-1], self.average_sensitivity_prob_history[-1]))
        else:
            pass
    
    def on_train_end(self, logs):
        self.plot_history()
        return
    
    def plot_history(self):
        plot_path = os.path.join(self.dir_path, self.plot_name + '.png')
        
        y_avg_snr = self.average_sensitivity_snr_history
        y_peak_snr = self.peak_sensitivity_snr_history
        y_avg_prob = self.average_sensitivity_prob_history
        y_peak_prob = self.peak_sensitivity_prob_history
        
        x = np.arange(self.interval, (len(y_avg_snr) + 1) * self.interval, self.interval)
        
        dpi = 96
        plt.figure(figsize=(1920.0/dpi, 1440.0/dpi), dpi=dpi)
        plt.rcParams.update({'font.size': 32, 'text.usetex': 'true'})
        
        plt.plot(x, y_avg_snr, label='Average SNR')
        plt.plot(x, y_peak_snr, label='Peak SNR')
        plt.plot(x, y_avg_prob, label='Average p-value')
        plt.plot(x, y_peak_prob, label='Peak p-value')
        plt.xlabel('Epoch')
        plt.ylabel('Sensitivity')
        plt.title('Sensitivity history')
        plt.legend()
        plt.grid()
        plt.savefig(plot_path)
        
        plt.cla()
        plt.clf()
        plt.close()
        return
