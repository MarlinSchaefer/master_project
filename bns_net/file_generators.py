import psutil
import keras
import numpy as np
import h5py
import os
import sys

def get_free_memory():
    mem = psutil.virtual_memory()
    return mem.available

class SliceSequence(object):
    def __init__(self, slices=None, start_stop=None, quiet=False):
        if slices is not None:
            assert len(slices) == len(start_stop)
        
        self.slices = [] if slices is None else slices
        self.start_stops = [] if start_stop is None else start_stop
        self.quiet = quiet
    
    @property
    def min_idx(self):
        if len(self.start_stops) > 0:
            return min([start for start, stop in self.start_stops])
        else:
            return np.inf
    
    @property
    def max_idx(self):
        if len(self.start_stops) > 0:
            return max([stop for start, stop in self.start_stops])
        else:
            return -np.inf
    
    def add_slice(self, data, index_range):
        contains = self._convert_to_data_index_range(index_range[0], index_range[1])
        if len(contains) == 0:
            self.slices.append(data)
            self.start_stops.append(index_range)
        else:
            #Handle overlaps
            start, stop = index_range
            to_remove = []
            add_lower = None
            add_upper = None
            for key, val in contains.items():
                slice_start, slice_stop = self.start_stops[key]
                if slice_start > start and slice_stop < stop:
                    to_remove.append([slice_start, slice_stop])
                elif slice_start > start:
                    add_lower = [key, start, slice_start]
                elif slice_stop < stop:
                    add_upper = [key, slice_stop, stop]
                elif slice_start < start and slice_stop > stop:
                    #There is a slice already that completely contains the slice that should be added.
                    return
            
            #remove unnecessary stuff
            for rm_start, rm_stop in to_remove:
                self.remove_slice(rm_start, rm_stop)
            
            if add_lower is not None and add_upper is not None:
                key_lower, start, slice_start = add_lower
                key_upper, slice_stop, stop = add_upper
                if (not key_lower == key_upper) and slice_start-start >= slice_stop-index_range[0]:
                    concat = np.concatenate([self.slices[key_upper], data[slice_stop-index_range[0]:slice_start-start], self.slices[key_lower]])
                    new_range = [self.start_stops[key_upper][0], self.start_stops[key_lower][1]]
                    rm_key = max(key_lower, key_upper)
                    keep_key = min(key_lower, key_upper)
                    self.remove_slice(self.start_stops[rm_key][0], self.start_stops[rm_key][1])
                    self.slices[keep_key] = concat
                    self.start_stops[keep_key] = new_range
            elif add_lower is not None:
                key, start, slice_start = add_lower
                self.slices[key] = np.concatenate([data[:slice_start-start], self.slices[key]])
                self.start_stops[key][0] = start
            elif add_upper is not None:
                key, slice_stop, stop = add_upper
                concat = np.concatenate([self.slices[key], data[slice_stop-index_range[0]:]])
                self.slices[key] = concat
                self.start_stops[key][1] = stop
            else:
                self.slices.append(data)
                self.start_stops.append(index_range)
            
        self._sort()
    
    def add_numpy_slice(self, array, start, stop):
        self.add_slice(array[start:stop], [start, stop])
    
    def remove_slice(self, start=None, stop=None):
        if start is None:
            start = self.min_idx
        if stop is None:
            stop = self.max_idx
        slice_idx = self._convert_to_data_index_range(start, stop)
        
        new_start_stops = []
        new_data = []
        for i, (start_idx, stop_idx) in slice_idx.items():
            p1_start = self.start_stops[i][0]
            p1_stop = self.start_stops[i][0] + start_idx
            p2_start = self.start_stops[i][0] + stop_idx
            p2_stop = self.start_stops[i][1]
            if p1_stop - p1_start > 0:
                new_start_stops.append([p1_start, p1_stop])
                new_data.append(self.slices[i][:start_idx])
            if p2_stop - p2_start > 0:
                new_start_stops.append([p2_start, p2_stop])
                new_data.append(self.slices[i][stop_idx:])
        for i in slice_idx.keys():
            self.slices.pop(i)
            self.start_stops.pop(i)
        self.slices = self.slices + new_data
        self.start_stops = self.start_stops + new_start_stops
        
        #TODO: Remove the function below and handle sorting in the add
        #      and remove functions.
        self._sort()
        
    def set_quiet(self, new):
        self.quiet = new
    
    def contains_index(self, idx):
        if len(self.start_stops) == 0:
            return False
        else:
            for start, stop in self.start_stops:
                if start <= idx and idx < stop:
                    return True
            return False
    
    def _convert_to_data_index(self, idx):
        if len(self.start_stops) == 0:
            return None, None
        else:
            for i, cont in enumerate(self.start_stops):
                start, stop = cont
                if start <= idx and idx < stop:
                    return i, idx - start
            return None, None
    
    def _sort(self):
        starts = [pt for pt, _  in self.start_stops]
        idx = np.argsort(starts)
        self.slices = [self.slices[i] for i in idx]
        self.start_stops = [self.start_stops[i] for i in idx]
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.get_range(start=idx.start, stop=idx.stop)
        slice_idx, data_idx = self._convert_to_data_index(idx)
        if slice_idx is None:
            if self.quiet:
                return None
            else:
                msg  = 'Index {} is not contained in this SliceSequence'
                raise IndexError(msg.format(idx))
        else:
            return self.slices[slice_idx][data_idx]
    
    def _convert_to_data_index_range(self, start, stop):
        if start > stop:
            raise ValueError('The beginning must be smaller than the end.')
        if start > self.max_idx or stop < self.min_idx:
            return {}
        
        best_slice_start = 0
        best_value_start = np.inf
        best_slice_stop = 0
        best_value_stop = -np.inf
        for i, (slice_start, slice_stop) in enumerate(self.start_stops):
            if slice_start <= start and start <= slice_stop:
                #Slice containing start
                best_slice_start = i
                best_value_start = start
            elif start < slice_start and slice_start <= stop:
                if slice_start < best_value_start:
                    #Slice found that is closer to the start index
                    best_slice_start = i
                    best_value_start = slice_start
            
            if slice_start <= stop and stop <= slice_stop:
                #Slice containing stop
                best_slice_stop = i
                best_value_stop = stop
            elif start <= slice_stop and slice_stop < stop:
                if best_value_stop < slice_stop:
                    #Slice found that is closer to the stop index
                    best_slice_stop = i
                    best_value_stop = slice_stop
        
        if np.isinf(best_value_start) or np.isinf(best_value_stop):
            return {}
        
        start_slice = best_slice_start
        stop_slice = best_slice_stop
        start_pos = best_value_start - self.start_stops[start_slice][0]
        stop_pos = best_value_stop - self.start_stops[stop_slice][0]
        
        if start_slice == stop_slice:
            return {start_slice: [start_pos, stop_pos]}
        
        ret = {}
        for i in range(start_slice, stop_slice + 1):
            if i == start_slice:
                ret[i] = [start_pos, self.start_stops[i][1] - self.start_stops[i][0]]
            elif i == stop_slice:
                ret[i] = [0, stop_pos]
            else:
                ret[i] = [0, len(self.slices[i])]
        return ret
    
    def get_range(self, start=None, stop=None):
        if start is None:
            if stop is not None and stop < self.min_idx:
                return np.array([])
            else:
                start = self.min_idx
        if stop is None:
            if start is not None and start > self.max_idx:
                return np.array([])
            else:
                stop = self.max_idx
        
        to_concat = []
        for i, (start, stop) in self._convert_to_data_index_range(start, stop).items():
            to_concat.append(self.slices[i][start:stop])
        if len(to_concat) == 0:
            return np.array([])
        return np.concatenate(to_concat)

class FileHandeler(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = None
    
    def __contains__(self, item):
        raise NotImplementedError
    
    def __enter__(self, mode='r'):
        raise NotImplementedError
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError
    
    def open(self, mode='r'):
        raise NotImplementedError
    
    def close(self):
        raise NotImplementedError

class MultiFileHandeler(object):
    def __init__(self, file_handelers=None, mode='r'):
        self._init_file_handelers(file_handelers)
        self.mode = mode
        self.input_shape = None #Shape the network expects as input
        self.output_shape = None #Shape the network expects as labels
    
    def __contains__(self, idx):
        contains = []
        index_split = self.split_index_to_groups(idx)
        for key, index in index_split.items():
            curr = False
            for file_handeler in self.file_handeler_groups[key]:
                if index in file_handeler:
                    curr = True
                    break
            contains.append(curr)
        return all(contains)
    
    def __enter__(self):
        for file_handeler in self.file_handelers:
            file_handeler.open(mode=self.mode)
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        for file_handeler in self.file_handelers:
            file_handeler.close()
    
    def __getitem__(self, idx):
        split_index = self.split_index_to_groups(idx)
        ret = {}
        for key, index in split_index.items():
            ret[key] = None
            for file_handeler in self.file_handeler_groups[key]:
                if index in file_handeler:
                    ret[key] = file_handeler[index]
        if any([val is None for val in ret.values()]):
            msg = 'The index {} was not found in any of the provided files.'
            raise IndexError(msg.format(idx))
        else:
            return self.format_return(ret)
    
    @property
    def file_handelers(self):
        return self.file_handeler_groups['all']
    
    def _init_file_handelers(self, file_handelers):
        if file_handelers is None:
            self.file_handeler_groups = {'all': []}
        elif isinstance(file_handelers, list):
            self.file_handeler_groups = {'all': file_handelers}
        elif isinstance(file_handelers, dict):
            self.file_handeler_groups = file_handelers
            self.file_handeler_groups['all'] = list(file_handelers.values())
    
    def add_file_handeler(self, file_handeler, group=None):
        if group is not None:
            if group in self.file_handeler_groups:
                self.file_handeler_groups[group].append(file_handeler)
            else:
                self.file_handeler_groups[group] = [file_handeler]
        self.file_handeler_groups['all'].append(file_handeler)
    
    def remove_file_handeler(self, file_handeler):
        for group in self.file_handeler_groups.values():
            if file_handeler in group:
                group.remove(file_handeler)
    
    def split_index_to_groups(self, idx):
        return {'all': idx}
    
    def format_return(self, inp):
        #Return value should be of form
        #input to network (np.array or list of np.array), labels for network (np.array or list of np.array), sample_weight (number or list of numbers)
        #The last one is optional and only required if the FileGenerator has use_sample_weights = True
        return inp['all']

class FileGenerator(keras.utils.Sequence):
    #TODO: Write a function that logs the indices used and the file-path
    def __init__(self, file_handeler, index_list, 
                 batch_size=32, shuffle=True, use_sample_weights=False):
        self.file_handeler = file_handeler
        self.index_list = index_list
        self.indices = np.arange(len(self.index_list), dtype=int)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_sample_weights = use_sample_weights
        self.on_epoch_end()
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(float(len(self.index_list)) / self.batch_size))
    
    def __getitem__(self, idx):
        if (idx + 1) * self.batch_size > len(self.indices):
            batch = self.indices[idx*self.batch_size:]
        else:
            batch = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_size = len(batch)
        if isinstance(self.file_handeler.input_shape, list):
            X = [np.zeros([batch_size] + list(shape)) for shape in self.file_handeler.input_shape]
        else:
            X = np.zeros([batch_size]+ list(self.file_handeler.input_shape))
        
        if isinstance(self.file_handeler.output_shape, list):
            Y = [np.zeros([batch_size] + list(shape)) for shape in self.file_handeler.output_shape]
        else:
            Y = np.zeros([batch_size] + list(self.file_handeler.output_shape))
        
        if self.use_sample_weights:
            if isinstance(self.file_handeler.output_shape, list):
                W = [np.zeros(batch_size) for _ in len(self.file_handeler.output_shape)]
            else:
                W = np.zeros(batch_size)
        
        for num, i in enumerate(batch):
            if self.use_sample_weights:
                inp, out, wei = self.file_handeler[self.index_list[i]]
                if isinstance(wei, list):
                    for part, w in zip(wei, W):
                        w[num] = part
                else:
                    for w in W:
                        w[num] = wei
            else:
                inp, out = self.file_handeler[self.index_list[i]]
            
            if isinstance(inp, list):
                for part, x in zip(inp, X):
                    x[num] = part
            else:
                X[num] = inp
        
            if isinstance(out, list):
                for part, y in zip(out, Y):
                    y[num] = part
            else:
                Y[num] = out
        
        if self.use_sample_weights:
            return X, Y, W
        else:
            return X, Y
        
            
        
