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
    def __init__(self, filters, frequency_low, frequency_high, activation=None, number_of_cycles=1, **kwargs):
        self.filters = filters
        self.frequency_low = float(frequency_low)
        self.frequency_high = float(frequency_high)
        self.activation = activations.get(activation)
        self.number_of_cycles = number_of_cycles
        self.dt = 0.5 / self.frequency_high
        self.rank = 1
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
        
        self.kernel_shape = (self.number_of_cycles * (int(np.floor(2 * self.frequency_high / self.frequency_low)) + 1),) + length[1:]
        print("Kernel shape: {}".format(self.kernel_shape))
        
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={-1: input_shape[-1]})
        
        super(FConv1D, self).build(input_shape)
    
    def call(self, x):
        np_kernel = np.zeros(self.kernel_shape, dtype=np.float32)
        
        np_freq, np_amp, np_phase = self.get_weights()
        
        print("Phase shape: {}".format(np_phase.shape))
        
        for i, freq in enumerate(np_freq[0]):
            for j, f in enumerate(freq):
                phase = np_phase[0][i][j]
                args = np.arange(phase, self.number_of_cycles / f + self.dt + phase, self.dt)
                #print("f: {}, upper_lim: {}".format(f, self.number_of_cycles / f))
                if len(args) == self.kernel_shape[0]:
                    args = np.arange(phase, self.number_of_cycles / f + phase, self.dt)
                #print("Shape of args: {}".format(args.shape))
                vals = np.sin(args)
                vals *= np_amp[0][i][j]
                vals = np.concatenate([vals, np.zeros(self.kernel_shape[0] - len(vals))])
                np_kernel[:,i,j] = vals
        
        kernel = tf.convert_to_tensor(np_kernel)
        
        #print("1: {}".format(kernel.shape))
        
        ones = K.ones(self.kernel_shape)
        
        a = self.frequencies * ones
        
        #print("shape of a: {}".format(a.shape))
        #print("Shape of weights: {}".format(self.frequencies.shape))
        #print("Shape of ones: {}".format(ones.shape))
        
        kernel *= a
        
        #print("2: {}".format(kernel.shape))
        
        b = self.phases * ones
        
        kernel += b
        
        #print("3: {}".format(kernel.shape))
        
        kernel = K.sin(kernel)
        
        #print("4: {}".format(kernel.shape))
        
        b = self.amplitudes * ones
        
        #print("5: {}".format(kernel.shape))
        
        kernel *= b
        
        ##Old non-working version
        #np_kernel = np.zeros(self.kernel_shape, dtype=np.float32)
        
        #np_freq, np_amp, np_phase = self.get_weights()
        
        #print("get_weights returns: {}".format(self.weights))
        
        #print("Real shape of numpy: {}".format(np_freq.shape))
        
        #print("Kernel shape: {}".format(self.kernel_shape))
        
        #for i, freq in enumerate(np_freq):
            #for j, f in enumerate(freq):
                #phase = np.array(np_phase)[i][j]
                #args = np.arange(phase, self.number_of_cycles / f + self.dt + phase, self.dt)
                ##print("f: {}, upper_lim: {}".format(f, self.number_of_cycles / f))
                #if len(args) == self.kernel_shape[0]:
                    #args = np.arange(phase, self.number_of_cycles / f + phase, self.dt)
                ##print("Shape of args: {}".format(args.shape))
                #vals = np.sin(args)
                #vals *= np.array(np_amp)[i][j]
                #vals = np.concatenate([vals, np.zeros(self.kernel_shape[0] - len(vals))])
                #np_kernel[:,j,i] = vals
        
        #kernel = tf.convert_to_tensor(np_kernel)
        
        #print("Kernel shape final: {}".format(kernel.shape))
        
        output = K.conv1d(x,
                          kernel,
                          padding='same',
                          data_format='channels_last')
        
        if self.activation is not None:
            return(self.activation(output))
        
        return(output)
    
    def compute_output_shape(self, input_shape):
        return(input_shape[:-1] + (self.filters, ))
