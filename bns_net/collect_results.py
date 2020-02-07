import os
import h5py
from aux_functions import get_store_path
import numpy as np

def getPercentage():
    resultPath = os.path.join(get_store_path(), 'long_data_2', 'results')
    collectedResultsPath = os.path.join(resultPath, 'collected_results.hf5')
    
    target = float(22 * len(np.arange(3, 202, 2)))
    
    current = 0.0
    
    with h5py.File(collectedResultsPath, 'r') as f:
        for k in f.keys():
            current += len(f[k].keys())
    
    return(current / target)

def partialPath(highLevel, lowLevel):
    resultPath = os.path.join(get_store_path(), 'long_data_2', 'results')
    return(os.path.join(resultPath, 'data_' + str(highLevel) + '_part_' + str(lowLevel)))

def tsPath(highLevel, lowLevel):
    return os.path.join(partialPath(highLevel, lowLevel), 'resulting_ts_data_' + str(highLevel) + '_part_'  + str(lowLevel) + '.hf5')

def triggerPath(highLevel, lowLevel):
    return(os.path.join(partialPath(highLevel, lowLevel), 'triggers_data_' + str(highLevel) + '_part_' + str(lowLevel) + '.hf5'))

def newFileExistsQ(highLevel, lowLevelList):
    for lowLevel in lowLevelList:
        if os.path.isfile(tsPath(highLevel, lowLevel)):
            return True
    return False

def writeToFile(fileObj, highLevel, lowLevel):
    if os.path.isfile(tsPath(highLevel, lowLevel)):
        high = fileObj.create_group(str(lowLevel))
        with h5py.File(tsPath(highLevel, lowLevel), 'r') as read:
            #Copy time series data
            ts = high.create_group('TimeSeries')
            
            ts.create_dataset('net_path', data=read['net_path'][()])
            ts.create_dataset('threshold_p-value', data=read['threshold_p-value'][()])
            ts.create_dataset('threshold_snr', data=read['threshold_snr'][()])
            
            ts_snr = ts.create_group('snrTimeSeries')
            ts_snr.create_dataset('data', data=read['snrTimeSeries/data'][:])
            ts_snr.create_dataset('sample_times', data=read['snrTimeSeries/sample_times'][:])
            ts_snr.create_dataset('delta_t', data=read['snrTimeSeries/delta_t'][()])
            ts_snr.create_dataset('epoch', data=read['snrTimeSeries/epoch'][()])
            
            ts_bool = ts.create_group('p-valueTimeSeries')
            ts_bool.create_dataset('data', data=read['p-valueTimeSeries/data'][:])
            ts_bool.create_dataset('sample_times', data=read['p-valueTimeSeries/sample_times'][:])
            ts_bool.create_dataset('delta_t', data=read['p-valueTimeSeries/delta_t'][()])
            ts_bool.create_dataset('epoch', data=read['p-valueTimeSeries/epoch'][()])
        
        with h5py.File(triggerPath(highLevel, lowLevel), 'r') as read:
            #Copy trigger data
            trig = high.create_group('Triggers')
            
            trig_snr = trig.create_group('snrTriggers')
            trig_snr.create_dataset('triggerTimes', data=read['snrTriggers/triggerTimes'][:])
            trig_snr.create_dataset('triggerValues', data=read['snrTriggers/triggerValues'][:])
            
            trig_bool = trig.create_group('p-valueTriggers')
            trig_bool.create_dataset('triggerTimes', data=read['p-valueTriggers/triggerTimes'][:])
            trig_bool.create_dataset('triggerValues', data=read['p-valueTriggers/triggerValues'][:])
            
            trig_comb = trig.create_group('combinedTriggers')
            trig_comb.create_dataset('triggerTimes', data=read['combinedTriggers/triggerTimes'][:])
            trig_comb.create_dataset('p-values', data=read['combinedTriggers/p-values'][:])
            trig_comb.create_dataset('snr-values', data=read['combinedTriggers/snr-values'][:])
    return

def main():
    resultPath = os.path.join(get_store_path(), 'long_data_2', 'results')
    collectedResultsPath = os.path.join(resultPath, 'collected_results.hf5')
    
    with h5py.File(collectedResultsPath, 'a') as f:
        highLevelIndices = list(np.arange(3, 202, 2, dtype=int))
        
        checkIndices = {}
        
        existingKeys = f.keys()
        
        existingKeys = [int(pt) for pt in existingKeys]
        
        #Check which files have not been put into the final one
        for idx in highLevelIndices:
            if idx in existingKeys:
                currIdx = f[str(idx)].keys()
                currIdx = [int(pt) for pt in currIdx]
                if len(currIdx) < 22:
                    checkIndices[idx] = []
                    for i in range(22):
                        if not i in currIdx:
                            checkIndices[idx].append(i)
            else:
                checkIndices[idx] = list(range(22))
        
        if len(checkIndices.keys()) == 0:
            print("File already complete.")
        else:
            for highLevel in sorted(checkIndices.keys()):
                print("Checking highLevel: {}".format(highLevel))
                lowLevelList = checkIndices[highLevel]
                if not str(highLevel) in f.keys() and newFileExistsQ(highLevel, lowLevelList):
                    f.create_group(str(highLevel))
                if newFileExistsQ(highLevel, lowLevelList):
                    for lowLevel in lowLevelList:
                        print("Got something new: ({}, {})".format(highLevel, lowLevel))
                        writeToFile(f[str(highLevel)], highLevel, lowLevel)
    
    print("The file is to {0:.2f}% complete.".format(100*getPercentage()))
    
    return

main()
