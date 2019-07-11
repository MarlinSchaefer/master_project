from keras import backend as K
from keras.layers import Layer
import numpy as np
from keras.constraints import Constraint, NonNeg
import tensorflow as tf
from keras import activations
from keras.initializers import RandomUniform
from keras.engine.base_layer import InputSpec

class MinMaxClip(Constraint):
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, w):
        #From keras.constraints.py function NonNeg
        #Sets the values of w to be non negative
        w *= K.cast(K.greater_equal(w, 0.), K.floatx())
        
        #From keras.constraints.py function MinMaxNorm
        #Clips the values of w to specific range
        desired = K.clip(w, self.min_val, self.max_val)
        w *= (desired / (K.epsilon() + w))
        return(w)

class FConv1D(Layer):
    def __init__(self, filters, frequency_low, frequency_high, activation=None, number_of_cycles=1, fill=True, window='hann', **kwargs):
        self.filters = filters
        self.frequency_low = float(frequency_low)
        self.frequency_high = float(frequency_high)
        self.activation = activations.get(activation)
        self.number_of_cycles = number_of_cycles
        self.dt = 0.5 / self.frequency_high
        self.rank = 1
        self.T = 1 / self.frequency_low
        self.T_LEN = int(np.floor(self.T / self.dt)) + 1
        self.fill = fill
        if not window in ['hann', None]:
            raise ValueError('Right now only Hanning and no windowing are supported.')
        else:
            self.window = window
        
        super(FConv1D, self).__init__(**kwargs)
    
    def build(self, input_shape):
        length = (1, input_shape[-1], self.filters)
        
        self.frequencies = self.add_weight(name='frequencies',
                                           shape=length,
                                           trainable=True,
                                           initializer=RandomUniform(minval=self.frequency_low, maxval=self.frequency_high),
                                           constraint=MinMaxClip(self.frequency_low, self.frequency_high),
                                           dtype=np.float32
                                           )
        
        self.amplitudes = self.add_weight(name='amplitudes',
                                          shape=length,
                                          trainable=True,
                                          initializer=RandomUniform(minval=1.0, maxval=2.0),
                                          constraint=NonNeg(),
                                          dtype=np.float32
                                          )
        
        self.phases = self.add_weight(name='phases',
                                      shape=length,
                                      trainable=True,
                                      initializer=RandomUniform(minval=0.0, maxval=2 * np.pi),
                                      constraint=MinMaxClip(0, 2*np.pi),
                                      dtype=np.float32
                                      )
        
        self.kernel_shape = (self.number_of_cycles * self.T_LEN,) + length[1:]
        if self.kernel_shape[0] > input_shape[1]:
            msg  = 'A low and high frequency cutoff of (f_low, f_high) of '
            msg += '({}, {}) '.format(self.frequency_low, self.frequency_high)
            msg += 'combined with a repition rate of {}'.format(self.number_of_cycles)
            msg += ' results in a kernel of length '
            msg += '{}. The maximum length '.format(self.kernel_shape[0])
            msg += 'however must be smaller or equal '
            msg += '{}. (input_shape[1])'.format(input_shape[1])
            raise ValueError(msg)
        
        #print("Kernel shape: {}".format(self.kernel_shape))
        
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={-1: input_shape[-1]})
        
        super(FConv1D, self).build(input_shape)
    
    def call(self, x):
        np_kernel = np.zeros(self.kernel_shape, dtype=np.float32)
        if not self.window == None:
            np_window = np.zeros(self.kernel_shape, dtype=np.float32)
        
        np_freq, np_amp, np_phase = self.get_weights()
        
        for i, freq in enumerate(np_freq[0]):
            for j, f in enumerate(freq):
                if not self.fill:
                    fT_len = int(np.floor(1 / (f * self.dt))) + 1
                    args = dt * np.arange(self.number_of_cycles * fT_len)
                    pad = np.zeros(self.T_LEN - len(args))
                    vals = np.concatenate([args, pad])
                    np_kernel[:,i,j] = vals
                    if not self.window == None:
                        np_window[:,i,j] = np.concatenate([np.hanning(len(args)), pad])
                else:
                    np_kernel[:,i,j] = self.dt * np.arange(self.number_of_cycles * self.T_LEN)
                    if not self.window == None:
                        np_window[:,i,j] =np.hanning(self.number_of_cycles * self.T_LEN)
                    
        kernel = tf.convert_to_tensor(np_kernel)
        
        ones = K.ones(self.kernel_shape)
        
        a = self.frequencies * ones
        
        kernel *= a
        
        b = self.phases * ones
        
        kernel += b
        
        kernel = K.sin(kernel)
        
        b = self.amplitudes * ones
        
        kernel *= b
        
        if not self.window == None:
            kernel += tf.convert_to_tensor(np_window)
        
        output = K.conv1d(x,
                          kernel,
                          padding='same',
                          data_format='channels_last')
        
        if self.activation is not None:
            return(self.activation(output))
        
        return(output)
    
    def compute_output_shape(self, input_shape):
        return(input_shape[:-1] + (self.filters, ))
