import h5py
import numpy as np
from aux_functions import get_store_path
import os

def get_results_path():
    return os.path.join(get_store_path(), 'long_data_2', 'results')

def get_collect_path():
    return os.path.join(get_results_path(), 'collected_results.hf5')

def get_stats_path():
    return os.path.join(get_results_path(), 'collected_stats.hf5')

def start_time(highLevel, lowLevel):
    #Convert highLevel to absolute index
    idxHigh = (highLevel - 3) / 2
    #Time per data-part
    t_part = 4096.
    #Trigger time addative, positions the triggers correctly
    trig_time_add = 65.5
    return((idxHigh * 22 + lowLevel) * t_part + trig_time_add)

def collectCombinedTriggers():
    clearList = list(np.arange(3, 202, 2))
    indices = {}
    with h5py.File(get_collect_path(), 'r') as f:
        intKeys = [int(pt) for pt in f.keys()]
        for i in clearList:
            if i in intKeys:
                indices[i] = [int(pt) for pt in f[str(i)].keys()]
                #Duration SNR TimeSeries: 4024s
                #Duration original TimeSeries: 4096s
        
        trigger_times = []
        snr_val = []
        bool_val = []
        for highLevel in sorted(indices.keys()):
            idxList = sorted(indices[highLevel])
            for lowLevel in idxList:
                add_time = start_time(highLevel, lowLevel)
                print("(highLevel, lowLevel, i) = ({}, {})".format(highLevel, lowLevel))
                for i, trig in enumerate(f[str(highLevel)][str(lowLevel)]['Triggers/combinedTriggers/triggerTimes'][:]):
                    print("(highLevel, lowLevel, i) = ({}, {}, {})".format(highLevel, lowLevel, i))
                    trigger_times.append(add_time + trig)
                    snr_val.append(f[str(highLevel)][str(lowLevel)]['Triggers/combinedTriggers/snr-values'][i])
                    bool_val.append(f[str(highLevel)][str(lowLevel)]['Triggers/combinedTriggers/p-values'][i])
        
        trigger_times = np.array(trigger_times)
        snr_val = np.array(snr_val)
        bool_val = np.array(bool_val)
        
        sort_idx = np.argsort(trigger_times)
        trigger_times = trigger_times[sort_idx]
        snr_val = snr_val[sort_idx]
        bool_val = bool_val[sort_idx]
        
        return (trigger_times, snr_val, bool_val)

def collectSnrTriggers():
    clearList = list(np.arange(3, 202, 2))
    indices = {}
    with h5py.File(get_collect_path(), 'r') as f:
        intKeys = [int(pt) for pt in f.keys()]
        for i in clearList:
            if i in intKeys:
                indices[i] = [int(pt) for pt in f[str(i)].keys()]
                #Duration SNR TimeSeries: 4024s
                #Duration original TimeSeries: 4096s
        
        trigger_times = []
        snr_val = []
        for highLevel in sorted(indices.keys()):
            idxList = sorted(indices[highLevel])
            for lowLevel in idxList:
                add_time = start_time(highLevel, lowLevel)
                print("(highLevel, lowLevel, i) = ({}, {})".format(highLevel, lowLevel))
                for i, trig in enumerate(f[str(highLevel)][str(lowLevel)]['Triggers/snrTriggers/triggerTimes'][:]):
                    print("(highLevel, lowLevel, i) = ({}, {}, {})".format(highLevel, lowLevel, i))
                    trigger_times.append(add_time + trig)
                    snr_val.append(f[str(highLevel)][str(lowLevel)]['Triggers/snrTriggers/triggerValues'][i])
        
        trigger_times = np.array(trigger_times)
        snr_val = np.array(snr_val)
        
        sort_idx = np.argsort(trigger_times)
        trigger_times = trigger_times[sort_idx]
        snr_val = snr_val[sort_idx]
        
        return (trigger_times, snr_val)

def collectBoolTriggers():
    clearList = list(np.arange(3, 202, 2))
    indices = {}
    with h5py.File(get_collect_path(), 'r') as f:
        intKeys = [int(pt) for pt in f.keys()]
        for i in clearList:
            if i in intKeys:
                indices[i] = [int(pt) for pt in f[str(i)].keys()]
                #Duration SNR TimeSeries: 4024s
                #Duration original TimeSeries: 4096s
        
        trigger_times = []
        bool_val = []
        for highLevel in sorted(indices.keys()):
            idxList = sorted(indices[highLevel])
            for lowLevel in idxList:
                add_time = start_time(highLevel, lowLevel)
                print("(highLevel, lowLevel, i) = ({}, {})".format(highLevel, lowLevel))
                for i, trig in enumerate(f[str(highLevel)][str(lowLevel)]['Triggers/p-valueTriggers/triggerTimes'][:]):
                    print("(highLevel, lowLevel, i) = ({}, {}, {})".format(highLevel, lowLevel, i))
                    trigger_times.append(add_time + trig)
                    bool_val.append(f[str(highLevel)][str(lowLevel)]['Triggers/p-value/triggerValues'][i])
        
        trigger_times = np.array(trigger_times)
        bool_val = np.array(bool_val)
        
        sort_idx = np.argsort(trigger_times)
        trigger_times = trigger_times[sort_idx]
        bool_val = bool_val[sort_idx]
        
        return (trigger_times, bool_val)

#Write function that returns dictionary {highLevel: [lowLevels]}
def availableData():
    ret = {}
    with h5py.File(get_collect_path(), 'r') as f:
        for k in f.keys():
            ret[int(k)] = [int(pt) for pt in f[k].keys()]
    return ret

def boundaries(highLevel, lowLevel):
    highIdx = (highLevel - 3) / 2
    low = (highIdx * 22 + lowLevel) * 4096.
    high = low + 4096.
    return [low, high]

#Write function that gives time boudnaries for such a dictionary
def getRanges(dic):
    #Functions expects dictionary as it is returned from availableData
    ret = []
    for highLevel in sorted(dic.keys()):
        for lowLevel in sorted(dic[highLevel]):
            ret.append(boundaries(highLevel, lowLevel))
    
    return np.array(ret)

def getValidInjections():
    ranges = getRanges(availableData())
    bool_values = []
    with h5py.File(get_stats_path(), 'r') as f:
        indices = []
        for i, t in enumerate(f['times'][:]):
            res = False
            for r in ranges:
                if t >= r[0] and t <= r[1]:
                    res = True
            bool_values.append(res)
            if res:
                indices.append(i)
        indices = np.array(indices)
        with h5py.File(os.path.join(get_results_path(), 'allowed_stats.hf5'), 'w') as g:
            for k in f.keys():
                dat = np.array(f[k][:])
                g.create_dataset(k, data=dat[indices])
    return

def clusterTriggerTimes(triggerTimes, time_span=0.5):
    indexRanges = []
    for i, pt in enumerate(triggerTimes):
        if i == 0:
            low = 0
            high = 0
            last = pt
        elif i == len(triggerTimes) - 1:
            up = low + 1 if high + 1 == low else high + 1
            indexRanges.append([low, up])
        else:
            #Cluster
            if pt - last > time_span:
                #Found new trigger
                up = low + 1 if high + 1 == low else high + 1
                indexRanges.append([low, up])
                low = high + 1
                high = low
            else:
                high += 1
            last = pt
                
    return indexRanges

def getTriggersAtThreshold(threshold, mode='snr', symmetricInjectionWindow=3., highCutoff=np.inf):
    """Get a list of SNR triggers above a threshold value.
    
    Arguments
    ---------
    threshold : float
        Value above which a trigger is considered
    mode : {optional, str}
        Mode of evaluation, either 'snr' or 'p-score'
    symmetricInjectionWindow : {optional, float}
        How big the window around a injection is, such that a trigger
        from the data is still considered to belong to the injection
    highCutoff : {optional, float}
        Value above which no trigger is generated
    
    Returns
    -------
    truePos : List
        List of triggers that where correctly identified
    falsePos : List
        List of tirggers that do not belong to an injection
    missed:
        List of injection times that were not found
    """
    if mode.lower() == 'snr':
        ts_data = 'TimeSeries/snrTimeSeries/data'
        ts_times = 'TimeSeries/snrTimeSeries/sample_times'
    elif mode.lower() == 'p-score':
        ts_data = 'TimeSeries/p-valueTimeSeries/data'
        ts_times = 'TimeSeries/p-valueTimeSeries/sample_times'
    else:
        raise NotImplementedError('Unsupported type {}'.format(mode))
    
    truePos = []
    falsePos = []
    missed = []
    with h5py.File(os.path.join(get_results_path(), 'allowed_stats.hf5'), 'r') as f:
        injectionTimes = f['times'][:]
    
    allowed = availableData()
    
    triggerTimes = []
    triggerVals = []
    with h5py.File(get_collect_path(), 'r') as f:
        for highLevel in sorted(allowed.keys()):
            for lowLevel in sorted(allowed[highLevel]):
                snr = f[str(highLevel)][str(lowLevel)][ts_data][:]
                times = f[str(highLevel)][str(lowLevel)][ts_times][:]
                times = boundaries(highLevel, lowLevel)[0] + 65.5 + times
                #Do I want the high cutoff here or do I veto it later?
                triggerIdx = np.where(snr > threshold)[0]
                triggerTimes = triggerTimes + list(times[triggerIdx])
                triggerVals = triggerVals + list(snr[triggerIdx])
    
    triggerTimes = np.array(triggerTimes)
    triggerVals = np.array(triggerVals)
    
    #Cluster the triggers, use the symmetricInjectionWindow as a time
    #during which the triggers belong to the same event
    clusters = clusterTriggerTimes(triggerTimes, time_span=1.0)
    
    #Find the actual trigger times and veto thos with too high values
    finalTimes = []
    finalVals = []
    for r in clusters:
        curr_cluster = triggerVals[r[0]:r[1]]
        allowed_indices = np.where(curr_cluster < highCutoff)[0]
        try:
            #If no point is below the upper threshold, this part will fail
            max_allowed_val = max(curr_cluster[allowed_indices])
            max_idx = np.where(curr_cluster == max_allowed_val)[0][0]
            finalTimes.append(triggerTimes[r[0] + max_idx])
            finalVals.append(triggerVals[r[0] + max_idx])
        except:
            pass
    
    #Find if trigger corresponds to actual injection
    for trig in finalTimes:
        found = False
        min_distance = np.inf
        for inj in injectionTimes:
            if abs(trig - inj) <= symmetricInjectionWindow:
                truePos.append((inj, trig))
                found = True
            else:
                if abs(trig - inj) < abs(min_distance):
                    min_distance = inj - trig
        if not found:
            falsePos.append((trig, min_distance))
    
    #Find missed injections
    foundInj = [pt[0] for pt in truePos]
    for inj in injectionTimes:
        if not inj in foundInj:
            missed.append(inj)
    
    return (truePos, falsePos, missed)

def getCombinedTriggersAtThreshold(threshold_snr, threshold_p, symmetricInjectionWindow=3., highCutoffSnr=np.inf, highCutoffP=np.inf):
    truePos = []
    falsePos = []
    missed = []
    with h5py.File(os.path.join(get_results_path(), 'allowed_stats.hf5'), 'r') as f:
        injectionTimes = f['times'][:]
    
    allowed = availableData()
    
    triggerTimes = []
    triggerVals = []
    with h5py.File(get_collect_path(), 'r') as f:
        for highLevel in sorted(allowed.keys()):
            for lowLevel in sorted(allowed[highLevel]):
                snr = f[str(highLevel)][str(lowLevel)][ts_data][:]
                times = f[str(highLevel)][str(lowLevel)][ts_times][:]
                times = boundaries(highLevel, lowLevel)[0] + 65.5 + times
                #Do I want the high cutoff here or do I veto it later?
                triggerIdx = np.where(snr > threshold)[0]
                triggerTimes = triggerTimes + list(times[triggerIdx])
                triggerVals = triggerVals + list(snr[triggerIdx])
    
    triggerTimes = np.array(triggerTimes)
    triggerVals = np.array(triggerVals)
    
    #Cluster the triggers, use the symmetricInjectionWindow as a time
    #during which the triggers belong to the same event
    clusters = clusterTriggerTimes(triggerTimes, time_span=1.0)
    
    #Find the actual trigger times and veto thos with too high values
    finalTimes = []
    finalVals = []
    for r in clusters:
        curr_cluster = triggerVals[r[0]:r[1]]
        allowed_indices = np.where(curr_cluster < highCutoff)[0]
        try:
            #If no point is below the upper threshold, this part will fail
            max_allowed_val = max(curr_cluster[allowed_indices])
            max_idx = np.where(curr_cluster == max_allowed_val)[0][0]
            finalTimes.append(triggerTimes[r[0] + max_idx])
            finalVals.append(triggerVals[r[0] + max_idx])
        except:
            pass
    
    #Find if trigger corresponds to actual injection
    for trig in finalTimes:
        found = False
        min_distance = np.inf
        for inj in injectionTimes:
            if abs(trig - inj) <= symmetricInjectionWindow:
                truePos.append((inj, trig))
                found = True
            else:
                if abs(trig - inj) < abs(min_distance):
                    min_distance = inj - trig
        if not found:
            falsePos.append((trig, min_distance))
    
    #Find missed injections
    foundInj = [pt[0] for pt in truePos]
    for inj in injectionTimes:
        if not inj in foundInj:
            missed.append(inj)
    
    return (truePos, falsePos, missed)
    
    
