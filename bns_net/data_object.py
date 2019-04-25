import numpy as np
import os
from run_net import get_store_path
import h5py

#TODO: Add load functions without output

class DataSet():
    def __init__(self, file_path='', retType='formatted'):
        self.set_file_path(file_path)
        
        if str(retType) == 'formatted':
            self.retType = str(retType)
        elif str(retType) == 'raw':
            self.retType = str(retType)
        else:
            raise ValueError("The retType has to be either 'formatted' or 'raw'. Your input was {}.".format(str(retType)))
        
        self.__init_indices()
        self.__init_data()
    
    def set_file_path(self, file_path):
        if type(file_path) == str:
            self.file_path = file_path
        else:
            raise ValueError('The provided file path has to be a string.')
    
    def get_file_path(self):
        return(self.file_path)
    
    @property
    def file_exists(self):
        try:
            with open(self.file_path, 'r') as FILE:
                return(True)
        except:
            return(False)
    
    def exists_q(self):
        return(self.file_exists)
    
    @property
    def training_data_shape(self):
        if self.file_exists:
            with h5py.File(self.file_path, 'r') as FILE:
                return(FILE['training']['train_data'].shape)
        else:
            return(None)
    
    @property
    def training_label_shape(self):
        if self.file_exists:
            with h5py.File(self.file_path, 'r') as FILE:
                return(FILE['training']['train_labels'].shape)
        else:
            return(None)
    
    @property
    def training_snr_shape(self):
        if self.file_exists:
            with h5py.File(self.file_path, 'r') as FILE:
                return(FILE['training']['train_snr_calculated'].shape)
        else:
            return(None)
    
    @property
    def testing_data_shape(self):
        if self.file_exists:
            with h5py.File(self.file_path, 'r') as FILE:
                return(FILE['testing']['test_data'].shape)
        else:
            return(None)
    
    @property
    def testing_label_shape(self):
        if self.file_exists:
            with h5py.File(self.file_path, 'r') as FILE:
                return(FILE['testing']['test_labels'].shape)
        else:
            return(None)
    
    @property
    def testing_snr_shape(self):
        if self.file_exists:
            with h5py.File(self.file_path, 'r') as FILE:
                return(FILE['testing']['test_snr_calculated'].shape)
        else:
            return(None)
    
    @property
    def shape(self):
        ret = {}
        ret['train_data'] = self.training_data_shape
        ret['train_labels'] = self.training_label_shape
        ret['train_snr_calculated'] = self.training_snr_shape
        ret['test_data'] = self.testing_data_shape
        ret['test_labels'] = self.testing_label_shape
        ret['test_snr_calculated'] = self.testing_snr_shape
        return(ret)
    
    def rel_ind_to_abs(self, ind, data_len):
        if not ind < 0:
            return ind
    
        return(data_len + ind)
    
    def get_raw_training_data(self, slice=None, point=None):
        return(get_raw_data('training', 'train_data', slice=slice, point=point))
    
    def get_raw_testing_data(self, slice=None, point=None):
        return(get_raw_data('testing', 'test_data', slice=slice, point=point))
    
    def get_raw_training_labels(self, slice=None, point=None):
        return(get_raw_data('training', 'train_labels', slice=slice, point=point))
    
    def get_raw_testing_labels(self, slice=None, point=None):
        return(get_raw_data('testing', 'test_labels', slice=slice, point=point))
    
    def get_raw_data(self, t, st, slice=None, point=None, retDiffOnly=False):
        #Sanity checks and defining parameters
        if t == 'training':
            if st in ['train_data', 'train_labels', 'train_snr_calculated']:
                s = st
            else:
                raise ValueError('st needs to be one of the following: {}'.format(['train_data', 'train_labels', 'train_snr_calculated']))
            
            if s == 'train_data':
                shape = self.training_data_shape
            elif s == 'train_labels':
                shape = self.training_label_shape
            elif s == 'train_snr_calculated':
                shape = self.training_snr_shape
        elif t == 'testing':
            if st in ['test_data', 'test_labels', 'test_snr_calculated']:
                s = st
            else:
                raise ValueError('st needs to be one of the following: {}'.format(['test_data', 'test_labels', 'test_snr_calculated']))
            
            if s == 'test_data':
                shape = self.testing_data_shape
            elif s == 'test_labels':
                shape = self.testing_label_shape
            elif s == 'test_snr_calculated':
                shape = self.testing_snr_shape
        else:
            raise ValueError("The paramaeter t must be eiter 'training' or 'testing'.")
        
        if not point == None:
            raise NotImplementedError('Right now only slices or the entire dataset can be loaded.')
            if type(point) == list:
                try:
                    ret = np.empty([len(point)] + list(shape)[1:])
                except IndexError:
                    ret = np.empty(len(point))
                
                try:
                    with h5py.File(self.file_path, 'r') as FILE:
                        for i, pt in enumerate(point):    
                            if not type(pt) == int:
                                raise ValueError('point has to be an integer or a list of integers.')
                            else:
                                ret[i] = FILE[t][s][pt]
                                self.loaded_indices.add(pt)
                        return(ret)
                except IOError:
                    raise ValueError('The file you are trying to access does not seem to exist.')
                
            elif type(point) == int:
                return
            else:
                raise ValueError('point has to be an integer or a list of integers.')
        
        if not slice == None:
            if not type(slice) in [list, tuple] or not len(slice) == 2:
                raise ValueError('slice needs to be a tuple or list of exactly 2 items.')
            
            try:
                with h5py.File(self.file_path) as FILE:
                    l = len(FILE[t][s])
                    low_ind = self.rel_ind_to_abs(slice[0], l)
                    
                    high_ind = self.rel_ind_to_abs(slice[1], l)
                    
                    if self.retType == 'formatted':
                        if retDiffOnly and not self.loaded_indices[t][s] == None:
                            part1 = None
                            part2 = None
                            if self.loaded_indices[t][s][0] > low_ind:
                                part1 = list(FILE[t][s][low_ind:self.loaded_indices[t][s][0]])
                            if self.loaded_indices[t][s][1] < high_ind:
                                part2 = list(list(FILE[t][s][self.loaded_indices[t][s][1]:high_ind]))
                            return([part1, part2])
                        else:
                            if retDiffOnly:
                                return([list(FILE[t][s][low_ind:high_ind]), None])
                            else:
                                return(list(FILE[t][s][low_ind:high_ind]))
                    
                    if self.loaded_indices[t][s] == None:
                        self.loaded_indices[t][s] = [low_ind, high_ind]
                        self.loaded_data[t][s] = list(FILE[t][s][low_ind:high_ind])
                        if retDiffOnly:
                            return([self.loaded_data[t][s], None])
                        else:
                            return(self.loaded_data[t][s])
                    
                    old_low = self.loaded_indices[t][s][0]
                    old_high = self.loaded_indices[t][s][1]
                    if low_ind < old_low:
                        if high_ind > old_high:
                            self.loaded_data[t][s] = list(FILE[t][s][low_ind:old_low]) + self.loaded_data[t][s] + list(FILE[t][s][old_high:high_ind])
                            self.loaded_indices[t][s][0] = low_ind
                            self.loaded_indices[t][s][1] = high_ind
                            
                            if retDiffOnly:
                                return([self.loaded_data[t][s][low_ind:old_ind], self.loaded_data[t][s][old_high:high_ind]])
                            else:
                                return(self.loaded_data[s][t])
                        else:
                            self.loaded_data[t][s] = list(FILE[t][s][low_ind:old_low]) + self.loaded_data[t][s]
                            self.loaded_indices[t][s][0] = low_ind
                            
                            if retDiffOnly:
                                return([self.loaded_data[t][s][low_ind:old_ind], None])
                            else:
                                return(self.loaded_data[s][t])
                    else:
                        if high_ind > old_high:
                            self.loaded_data[t][s] = self.loaded_data[t][s] + list(FILE[t][s][old_high:high_ind])
                            self.loaded_indices[t][s][1] = high_ind
                            
                            if retDiffOnly:
                                return([None, self.loaded_data[t][s][old_high:high_ind]])
                            else:
                                return(self.loaded_data[s][t])
                        else:
                            self.loaded_data[t][s] = self.loaded_data[t][s]
                            
                            if retDiffOnly:
                                return([None, None])
                            else:
                                return(self.loaded_data[s][t])
            except IOError:
                raise ValueError('The file you are trying to access does not seem to exist.')
        
        try:
            with h5py.File(self.file_path, 'r') as FILE:
                low_ind = 0
                high_ind = len(FILE[t][s]) - 1
                
                if not self.loaded_indices[t][s] == None:
                    old_low = self.loaded_indices[t][s][0]
                else:
                    old_low = -np.inf
                if not self.loaded_indices[t][s] == None:
                    old_high = self.loaded_indices[t][s][1]
                else:
                    old_high = np.inf
                part1 = None
                part2 = None
                
                if self.loaded_indices[t][s] == None:
                    part1 = list(FILE[t][s][:])
                    if self.retType == 'raw':
                        self.loaded_indices[t][s] = [low_ind, high_ind]
                
                if low_ind < old_low:
                    part1 = list(FILE[t][s][low_ind:old_low])
                    if self.retType == 'raw':
                        self.loaded_indices[t][s][0] = low_ind
                
                if high_ind > old_high:
                    part2 = list(FILE[t][s][old_high:high_ind])
                    if self.retType == 'raw':
                        self.loaded_indices[t][s][1] = high_ind
                
                if self.retType == 'raw':
                    part1n = part1 if not part1 == None else []
                    part2n = part2 if not part2 == None else []
                    if not self.loaded_data[t][s] == None:
                        self.loaded_indices[t][s] = [low_ind, high_ind]
                        self.loaded_data[t][s] = part1n + self.loaded_data[t][s] + part2n
                    else:
                        self.loaded_indices[t][s] = [low_ind, high_ind]
                        self.loaded_data[t][s] = part1n + part2n
                
                if retDiffOnly:
                    return([part1, part2])
                else:
                    if part1 == None:
                        part1 = []
                    if part2 == None:
                        part2 = []
                    if abs(old_low) == np.inf or abs(old_high) == np.inf:
                        return(part1 + part2)
                    else:
                        return(part1 + list(FILE[t][s][old_low:old_high]) + part2)
        except IOError:
            raise ValueError('The file you are trying to access does not seem to exist.')
    
    def format_data_segment(self, data_segment):
        return(data_segment)
    
    def format_label_segment(self, label_segment):
        return(label_segment)
    
    def format_snr_segment(self, snr_segment):
        return(snr_segment)
    
    def get_formatted_data(self, t, st, slice=None, point=None, retDiffOnly=False):
        print("Now formatting {}: {}".format(t, st))
        #Sanity checks and defining parameters
        if t == 'training':
            if st in ['train_data', 'train_labels', 'train_snr_calculated']:
                s = st
            else:
                raise ValueError('st needs to be one of the following: {}'.format(['train_data', 'train_labels', 'train_snr_calculated']))
            
            if s == 'train_data':
                shape = self.training_data_shape
            elif s == 'train_labels':
                shape = self.training_label_shape
            elif s == 'train_snr_calculated':
                shape = self.training_snr_shape
        elif t == 'testing':
            if st in ['test_data', 'test_labels', 'test_snr_calculated']:
                s = st
            else:
                raise ValueError('st needs to be one of the following: {}'.format(['test_data', 'test_labels', 'test_snr_calculated']))
            
            if s == 'test_data':
                shape = self.testing_data_shape
            elif s == 'test_labels':
                shape = self.testing_label_shape
            elif s == 'test_snr_calculated':
                shape = self.testing_snr_shape
        else:
            raise ValueError("The paramaeter t must be eiter 'training' or 'testing'.")
        
        if self.retType == 'formatted':
            #Handle loading data into storage and retrieving from there.
            if not point == None:
                raise NotImplementedError('Right now only slices or the entire dataset can be loaded.')
        
            if not slice == None:
                if not type(slice) in [list, tuple] or not len(slice) == 2:
                    raise ValueError('The slice option needs to be a list or tuple of length 2.')
            
                low_ind = self.rel_ind_to_abs(slice[0], self.shape[s][0])
                high_ind = self.rel_ind_to_abs(slice[1], self.shape[s][0])
            else:
                low_ind = 0
                high_ind = self.shape[s][0] - 1
            
            if s in ['train_data', 'test_data']:
                d = self.get_raw_data(t, s, slice=slice, point=point, retDiffOnly=True)
                if d[0] == None:
                    d[0] = []
                if d[1] == None:
                    d[1] = []
                d = [self.format_data_segment(d[0]), self.format_data_segment(d[1])]
            elif s in ['train_labels', 'test_labels']:
                d = self.get_raw_data(t, s, slice=slice, point=point, retDiffOnly=True)
                if d[0] == None:
                    d[0] = []
                if d[1] == None:
                    d[1] = []
                d = [self.format_label_segment(d[0]), self.format_label_segment(d[1])]
            elif s in ['train_snr_calculated', 'test_snr_calculated']:
                d = self.get_raw_data(t, s, slice=slice, point=point, retDiffOnly=True)
                if d[0] == None:
                    d[0] = []
                if d[1] == None:
                    d[1] = []
                d = [self.format_snr_segment(d[0]), self.format_snr_segment(d[1])]
            else:
                raise RuntimeError('Got an unsupported kind of data handle: {}'.format(s))
                
            self.join_formatted(t, s, d[0], d[1])
            
            self.loaded_indices[t][s] = [low_ind, high_ind]
            
            return(self.loaded_data[t][s][low_ind - self.loaded_indices[t][s][0]:high_ind - self.loaded_indices[t][s][0]])
            
        else:
            if s in ['train_data', 'test_data']:
                return(self.format_data_segment(self.get_raw_data(t, s, slice=slice, point=point)))
            elif s in ['train_labels', 'test_labels']:
                return(self.format_label_segment(self.get_raw_data(t, s, slice=slice, point=point)))
            elif s in ['train_snr_calculated', 'test_snr_calculated']:
                return(self.format_snr_segment(self.get_raw_data(t, s, slice=slice, point=point)))
            else:
                raise RuntimeError('Got an unsupported kind of data handle: {}'.format(s))
    
    @property
    def loaded_train_data(self):
        return(self.loaded_data['training']['train_data'])
    
    @property
    def loaded_train_labels(self):
        return(self.loaded_data['training']['train_labels'])
    
    @property
    def loaded_train_snr(self):
        return(self.loaded_data['training']['train_snr_calculated'])
    
    @property
    def loaded_test_data(self, slice=None):
        return(self.loaded_data['testing']['test_data'])
    
    @property
    def loaded_test_labels(self, slice=None):
        return(self.loaded_data['testing']['test_labels'])
    
    @property
    def loaded_test_snr(self, slice=None):
        return(self.loaded_data['testing']['test_snr_calculated'])
    
    #def get_set(self, slice=None):
        #print("Called get set")
        #if self.retType == 'formatted':
            #return(((self.get_formatted_data('training', 'train_data', slice=slice),
                    #self.get_formatted_data('training', 'train_labels', slice=slice),
                    #self.get_formatted_data('training', 'train_snr_calculated', slice=slice)),
                    #(self.get_formatted_data('testing', 'test_data', slice=slice),
                     #self.get_formatted_data('testing', 'test_labels', slice=slice),
                     #self.get_formatted_data('testing', 'test_snr_calculated', slice=slice)))
            #)
        #elif self.retType == 'raw':
            #return(((self.get_raw_data('training', 'train_data', slice=slice),
                    #self.get_raw_data('training', 'train_labels', slice=slice),
                    #self.get_raw_data('training', 'train_snr_calculated', slice=slice)),
                    #(self.get_raw_data('testing', 'test_data', slice=slice),
                     #self.get_raw_data('testing', 'test_labels', slice=slice),
                     #self.get_raw_data('testing', 'test_snr_calculated', slice=slice)))
            #)
    
    def get_set(self, slice=None):
        print("Called get set")
        if self.retType == 'formatted':
            print("Pre training data")
            tr_d = self.get_formatted_data('training', 'train_data', slice=slice)
            print("Pre training labels")
            tr_l = self.get_formatted_data('training', 'train_labels', slice=slice)
            print("Pre training SNR")
            tr_s = self.get_formatted_data('training', 'train_snr_calculated', slice=slice)
            print("Pre testing data")
            te_d = self.get_formatted_data('testing', 'test_data', slice=slice)
            print("Pre testing labels")
            te_l = self.get_formatted_data('testing', 'test_labels', slice=slice)
            print("Pre testing SNR")
            te_s = self.get_formatted_data('testing', 'test_snr_calculated', slice=slice)
            print("Loaded everything, now returning")
            return(((tr_d, tr_l, tr_s), (te_d, te_l, te_s)))
        elif self.retType == 'raw':
            return(((self.get_raw_data('training', 'train_data', slice=slice),
                    self.get_raw_data('training', 'train_labels', slice=slice),
                    self.get_raw_data('training', 'train_snr_calculated', slice=slice)),
                    (self.get_raw_data('testing', 'test_data', slice=slice),
                     self.get_raw_data('testing', 'test_labels', slice=slice),
                     self.get_raw_data('testing', 'test_snr_calculated', slice=slice)))
            )
    
    def __init_data(self):
        self.loaded_data = {'training': {}, 'testing': {}}
        self.loaded_data['training']['train_data'] = None
        self.loaded_data['training']['train_labels'] = None
        self.loaded_data['training']['train_snr_calculated'] = None
        self.loaded_data['testing']['test_data'] = None
        self.loaded_data['testing']['test_labels'] = None
        self.loaded_data['testing']['test_snr_calculated'] = None
    
    def __init_indices(self):
        self.loaded_indices = {'training': {}, 'testing': {}}
        self.loaded_indices['training']['train_data'] = None
        self.loaded_indices['training']['train_labels'] = None
        self.loaded_indices['training']['train_snr_calculated'] = None
        self.loaded_indices['testing']['test_data'] = None
        self.loaded_indices['testing']['test_labels'] = None
        self.loaded_indices['testing']['test_snr_calculated'] = None
    
    def unload_all(self):
        del self.loaded_indices['training']['train_data']
        del self.loaded_indices['training']['train_labels']
        del self.loaded_indices['training']['train_snr_calculated']
        del self.loaded_indices['testing']['test_data']
        del self.loaded_indices['testing']['test_labels']
        del self.loaded_indices['testing']['test_snr_calculated']
        
        del self.loaded_data['training']['train_data']
        del self.loaded_data['training']['train_labels']
        del self.loaded_data['training']['train_snr_calculated']
        del self.loaded_data['testing']['test_data']
        del self.loaded_data['testing']['test_labels']
        del self.loaded_data['testing']['test_snr_calculated']
        
        del self.loaded_indices
        del self.loaded_data
        
        self.__init_indices()
        self.__init_data()
    
    def join_formatted(self, t, s, part1, part2):
        if part1 == None:
            part1 = []
        if part2 == None:
            part2 = []
        if self.loaded_data[t][s] == None:
            self.loaded_data[t][s] = part1 + part2
        else:
            self.loaded_data[t][s] = part1 + self.loaded_data[t][s] + part2
        
        if self.loaded_data[t][s] == []:
            self.loaded_data[t][s] = None
        
        return
    
    def get_file_properties(self):
        ret = {}
        with h5py.File(self.file_path, 'r') as FILE:
            for k in FILE['parameter_space'].keys():
                ret[str(k)] = FILE['parameter_space'][k].value
        return(ret)
        
