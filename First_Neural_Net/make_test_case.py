from pycbc.waveform import get_td_waveform
import pycbc.noise
import pycbc.psd
import random
from pycbc.types.timeseries import TimeSeries
import matplotlib.pyplot as plt

"""
Returns a signal and a boolean value to signify, wether or not
an actual signal is present in the data.

Args:
    -(string)approximant: The approximant to use
    -(float)time_length: The duration for which the signal
                         should last
    -(float)mass1: primary mass
    -(float)mass2: secondary mass
    -(float)delta_t: sample frequency
    -(int)f_lower: lower frequency cutoff
    -(int)distance: Distance to the merger event in mega Parsecs

Ret:
    -(tuple): first entry pycbc.types.TimeSeries object,
              containing the signal, second entry GW-Signal, third entry boolean signifying wether or not a
              GW-signal is present in the data
"""
class test_case():
    def __init__(self,approximant,time_length,mass1,mass2,delta_t,f_lower,distance):
        self.parameters = self.set_parameters(approximant,time_length,mass1,mass2,delta_t,f_lower,distance)
        self.psd = self.create_psd()
        self.noise = self.create_noise()
        self.gw_present = bool(round(random.random()))
        self.gw = self.create_gw()
        self.noise._epoch = self.gw._epoch
        self.total = self.noise + self.gw
    
    def set_parameters(self,approximant,time_length,mass1,mass2,delta_t,f_lower,distance):
        paras = {}
        paras['approximant'] = approximant
        paras['time_length'] = time_length
        paras['mass1'] = mass1
        paras['mass2'] = mass2
        paras['delta_t'] = delta_t
        paras['f_lower'] = f_lower
        paras['distance'] = distance
        paras['delta_f'] = 1.0 / time_length
        paras['f_len'] = int(2 / (delta_t * paras['delta_f'])) + 1
        paras['t_samples'] = int(time_length / delta_t)
        return(paras)
    
    def create_psd(self):
        return(pycbc.psd.aLIGOZeroDetHighPower(self.parameters['f_len'], self.parameters['delta_f'], self.parameters['f_lower']))
    
    def create_noise(self):
        return(pycbc.noise.noise_from_psd(self.parameters['t_samples'], self.parameters['delta_t'], self.psd, seed=random.randint(1,100)))
    
    def create_gw(self):
        if self.gw_present:
            #Generate signal
            hp, hc = get_td_waveform(approximant=self.parameters['approximant'],mass1=self.parameters['mass1'],mass2=self.parameters['mass2'],delta_t=self.parameters['delta_t'],f_lower=self.parameters['f_lower'],distance=self.parameters['distance'])
            strain = self.detector_response(hp,hc)
            
            #Put GW somewhere in the noise and append/prepend the right amount of zeros
            t_start = random.randint(0,self.parameters['t_samples']-len(strain)-1)
            strain.prepend_zeros(t_start)
            strain.append_zeros(self.parameters['t_samples']-len(strain))
            return(strain)
        else:
            gw = TimeSeries([0],delta_t=self.parameters['delta_t'])
            gw.append_zeros(int(self.parameters['time_length'] / self.parameters['delta_t']) - 1)
            return(gw)
    
    """
    TODO: Implement correctly
    """
    def detector_response(self,hp,hc):
        return(hp)
    
    def plot(self):
        if self.gw_present:
            plt.plot(self.total.sample_times, self.total,label='Strain')
            plt.plot(self.gw.sample_times, self.gw, label='GW-Signal')
            plt.grid()
            plt.legend()
            plt.show()
        else:
            plt.plot(self.total.sample_times, self.total, label='Strain')
            plt.grid()
            plt.legend()
            plt.show()
