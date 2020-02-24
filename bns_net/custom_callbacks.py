import keras.backend as K
import keras
from keras.callbacks import  Callback
import numpy as np
import os
import matplotlib.pyplot as plt
import json
from progress_bar import progress_tracker
import warnings

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

class TestInputForSimplicity(Callback):
    def __init__(self, training_generator, validation_generator, 
                 test_model=None, interval=1, verbose=1, on_epoch=None,
                 loss='mse', optimizer='adam', test_epochs=1,
                 warning_threshold='real_model'):
        """Callback that trains a very simple network on the data to
        check if the input is wrong/has a very simple pattern.
        
        The chances that a very simple network will learn to pick up on
        specific details quickly are high(?). If this callback performs
        well on the training generator something might be fishy.
        
        Arguments
        ---------
        training_generator : keras.utils.Sequence
            The generator that is used to train the main model.
        validation_generator : keras.utils.Sequence
            The generator that is used to validate the main model.
        test_model : {keras.models.Model or None, None}
            The simple model that should be used to validate the
            training- and validation generator. It is supposed to be
            very simple and train quickly. If set to None a very simple
            network will be setup with the same inputs and outputs as
            the main model. See self._init_test_model for the specific
            model that will be used.
        interval : {int, 1}
            How many epochs should pass between invocations of this
            callback. If on_epoch is not None this option will be
            ignored.
        verbose : {int, 1}
            Whether or not and how much to print to the console.
            (0: No prints, 1: Print only final results, 2: Print full
            training)
        on_epoch : {int or None, None}
            If this option is not None the test will only be carried out
            at the specified epoch.
        loss : {keras loss, 'mse'}
            The loss function to be used for training. This option will
            be passed directly to the fit_generator function of keras
            models. Refer to the documentation of fit_generator for more
            details.
        optimizer : {keras optimizer, 'adam'}
            The optimizer function to be used for training. This option
            is passed directly to the fit_generator function of keras
            models. Refer to the documentation of fit_generator for more
            details.
        test_epochs : {int, 1}
            The number of epochs to train the test_model for before
            reporting results.
        warning_threshold : {str or float or iterable of floats, 'real_model'}
            A condition of when to warn the user that something seems
            off with the inputs. It warns once the loss of the
            test_model on the validation data falls below a given
            threshold.
            If this option is a string it has to be one of:
            ['real_model'].
            For further explanation see the Notes section. If a single
            float is given the callback will warn the user if the total
            validation loss of the test_model falls below this value.
            If a list of numbers is provided they are expected to match
            up with the length of losses and metrics as returned by the
            evaluate_generator function of the test_model. (If no custom
            test_model was provided it will have the same outputs as the
            real model) If the length don't match the callback will send
            out a warning anyways.
        
        Notes
        -----
        -Different string options for warning_threshold:
            +'real_model': Warn once the total loss of the test_model
                           trained on the training data falls below the
                           validation loss of the real model. (The one
                           to which this callback was applied.)
        """
        super(TestInputForSimplicity, self).__init__()
        self.training_generator = training_generator
        self.validation_generator = validation_generator
        self.optimizer = optimizer
        self.loss = loss
        if test_model is not None:
            self.initialized = True
            self.test_model = test_model
            self.model.compile(optimizer=self.optimizer, loss=self.loss)
        else:
            self.initialized = False
        if on_epoch is None:
            self.interval = interval
            self.on_epoch = np.inf
        else:
            self.interval = np.inf
            self.on_epoch = on_epoch
        self.verbose = verbose
        if self.verbose == 0 or self.verbose == 1:
            self.fit_verbose = 0
        elif self.verbose == 2:
            self.fit_verbose = 1
        self.test_epochs = test_epochs
        self.warning_threshold = warning_threshold
        self.warned_on_epochs = []
    
    def _init_test_model(self):
        def reduce_input(inp):
            input_shape = [pt.value for pt in inp.shape[1:]]
            p1 = keras.layers.Flatten()(inp) #Out: sum(input_shape)
            p2 = keras.layers.Reshape((-1, 1))(p1) #Out: (sum(input_shape), 1)
            p3 = keras.layers.Conv1D(filters=16, kernel_size=16, padding='same', activation='relu')(p2) #Out: (sum(input_shape), 16)
            p4 = keras.layers.Conv1D(filters=1, kernel_size=1, activation='relu')(p3) #Out: (sum(input_shape), 1)
            p5 = keras.layers.AveragePooling1D(np.prod(input_shape) // 16)(p4) #Out: (16 or 17, 1)
            p6 = keras.layers.Flatten()(p5) #Out: 16 or 17
            p7 = keras.layers.Dense(1)(p6) #Out: 1
            p8 = keras.layers.Reshape((1, 1))(p7) #Out: (1, 1)
            return p8
        
        def expand_output(inp, output_shape):
            expand = keras.layers.Dense(np.prod(output_shape))(inp)
            ret = keras.layers.Reshape(output_shape)(expand)
            return ret
            
        inputs = self.model.inputs
        output_shapes = self.model.output_shape
        if not isinstance(output_shapes, list):
            output_shapes = [output_shapes]
        conc = keras.layers.concatenate([reduce_input(layer) for layer in inputs])
        flatten_1 = keras.layers.Flatten()(conc)
        dense_1 = keras.layers.Dense(1)(flatten_1)
        outputs = [expand_output(dense_1, shape[1:]) for shape in output_shapes]
        self.test_model = keras.models.Model(inputs=inputs, outputs=outputs)
        self.test_model.compile(loss=self.loss, optimizer=self.optimizer)
        self.initialized = True
        return
    
    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.interval == 0 or (epoch + 1) == self.on_epoch:
            if not self.initialized:
                self._init_test_model()
            
            #Don't need a validation step if no output is printed
            training_msg = "Training test model"
            if self.verbose > 1:
                print(training_msg)
                self.test_model.fit_generator(self.training_generator,
                                              validation_data=self.validation_generator,
                                              verbose=self.fit_verbose,
                                              epochs=self.test_epochs)
            else:
                print(training_msg)
                self.test_model.fit_generator(self.training_generator,
                                              verbose=self.fit_verbose,
                                              epochs=self.test_epochs)
            
            val_loss = self.test_model.evaluate_generator(self.validation_generator)
            
            if isinstance(self.warning_threshold, float):
                test_comp = np.array([val_loss[0]])
                true_comp = np.array([self.warning_threshold])
            elif isinstance(self.warning_threshold, str):
                if self.warning_threshold == 'real_model':
                    test_comp = np.array([val_loss[0]])
                    true_comp = np.array([logs['val_loss']])
                else:
                    msg  = 'Unknown warning_threshold '
                    msg += '{}.\n'.format(self.warning_threshold)
                    msg += 'test_model loss is given by: {}'.format(val_loss)
                    warnings.warn(msg, RuntimeWarning)
            else:
                try:
                    if len(self.warning_threshold) == len(val_loss):
                        test_comp = np.array(val_loss)
                        true_comp = np.array(self.warning_threshold)
                    else:
                        msg  = 'The list of provided threshold values '
                        msg += 'is not of the correct length. Using the'
                        msg += 'first provided value to compare against '
                        msg += 'the total validation loss of the '
                        msg += 'test_model. The total validation loss '
                        msg += 'is given by: {}'.format(val_loss)
                        warnings.warn(msg, RuntimeWarning)
                        test_comp = np.array([val_loss[0]])
                        true_comp = np.array([self.warning_threshold[0]])
                except TypeError:
                    msg  = 'The option \'warning_threshold\' '
                    msg += '{} had an '.format(self.warning_threshold)
                    msg += 'unsupported type. The check will pass if '
                    msg += 'the test_model did not produce a nan loss.'
                    warning.warn(msg, RuntimeWarning)
                    test_comp = np.array([val_loss[0]])
                    true_comp = np.array([-np.inf])
            
            if any(test_comp < true_comp):
                msg  = 'The test_model loss given by {}'.format(test_comp)
                msg += ' was at some position smaller than the given limit'
                msg += ', which is given by {}.'.format(true_comp)
                warnings.warn(msg, RuntimeWarning)
                self.warned_on_epochs.append(epoch + 1)
            else:
                if self.verbose > 0:
                    print("Test passed for the provided test_model.")
        else:
            pass
    
    def on_train_end(self, *args, **kwargs):
        if self.verbose > 0:
            if len(self.warned_on_epochs) == 0:
                print("All checks passed for the test model.")
            elif len(self.warned_on_epochs) == 1:
                print("Warned on epoch {}".format(self.warned_on_epochs[0]))
            else:
                print("Warned on epochs {}".format(self.warned_on_epochs))
    
    def get_test_model(self):
        if self.initialized:
            return self.test_model
        else:
            return None

class PrintLogs(Callback):
    def on_epoch_end(self, epoch, logs={}):
        print("Epoch: {}".format(epoch))
        print("Logs: {}".format(logs))
