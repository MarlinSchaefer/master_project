import h5py
import numpy as np
from aux_functions import get_store_path
import os
from progress_bar import progress_tracker
from pycbc.sensitivity import volume_montecarlo
import sys

def get_results_path():
    return os.path.join(get_store_path(), 'long_data_2', 'results')

def get_collect_path():
    return os.path.join(get_results_path(), 'collected_results.hf5')

def get_stats_path():
    return os.path.join(get_results_path(), 'collected_stats.hf5')

def get_snr_stats_path():
    return os.path.join(get_results_path(), 'collected_stats_snr.hf5')

def start_time(highLevel, lowLevel):
    #Convert highLevel to absolute index
    idxHigh = (highLevel - 3) / 2
    #Time per data-part
    t_part = 4096.
    #Trigger time addative, positions the triggers correctly
    trig_time_add = 65.5
    return((idxHigh * 22 + lowLevel) * t_part + trig_time_add)

def mchirp(m1, m2):
    return (m1 * m2) ** (3. / 5.) / (m1 + m2) ** (1. / 5.)

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

def time_to_file_index(time):
    highLevel = 2 * int(time / (4096 * 22)) + 3
    lowLevel = int((time % (4096 * 22)) / 4096)
    return highLevel, lowLevel

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

def getTriggersAtThreshold(threshold, mode='snr', symmetricInjectionWindow=3., highCutoff=np.inf, clusterTime=1.0):
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
    clusterTime : {optional, float}
        Time two individual triggers are allowed to be separated by and
        still belong to the same cluster. If this value is smaller than
        0.25 all triggers are treated individually and no clustering
        takes place.
    
    Returns
    -------
    truePos : List
        List of triggers that where correctly identified. Each entry is
        a tuple containing (injection time, recovered time, recovered
        value at that time)
    falsePos : List
        List of tirggers that do not belong to an injection. Each entry
        is a tuple containing (recovered time, distance to nearest
        injection, value at recovered time)
    missed:
        List of injection times that were not found
    """
    if mode.lower() == 'snr':
        ts_data = 'TimeSeries/snrTimeSeries/data'
        ts_times = 'TimeSeries/snrTimeSeries/sample_times'
        secondary_data = 'TimeSeries/p-valueTimeSeries/data'
    elif mode.lower() == 'p-score':
        ts_data = 'TimeSeries/p-valueTimeSeries/data'
        ts_times = 'TimeSeries/p-valueTimeSeries/sample_times'
        secondary_data = 'TimeSeries/snrTimeSeries/data'
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
    triggerValsSecondary = []
    with h5py.File(get_collect_path(), 'r') as f:
        for highLevel in sorted(allowed.keys()):
            for lowLevel in sorted(allowed[highLevel]):
                primary = f[str(highLevel)][str(lowLevel)][ts_data][:]
                secondary = f[str(highLevel)][str(lowLevel)][secondary_data][:]
                times = f[str(highLevel)][str(lowLevel)][ts_times][:]
                times = boundaries(highLevel, lowLevel)[0] + 65.5 + times
                #Do I want the high cutoff here or do I veto it later?
                triggerIdx = np.where(primary > threshold)[0]
                triggerTimes = triggerTimes + list(times[triggerIdx])
                triggerVals = triggerVals + list(primary[triggerIdx])
                triggerValsSecondary = triggerValsSecondary + list(secondary[triggerIdx])
    
    triggerTimes = np.array(triggerTimes)
    triggerVals = np.array(triggerVals)
    triggerValsSecondary = np.array(triggerValsSecondary)
    
    #Cluster the triggers, use the symmetricInjectionWindow as a time
    #during which the triggers belong to the same event
    clusters = clusterTriggerTimes(triggerTimes, time_span=clusterTime)
    
    #Find the actual trigger times and veto thos with too high values
    finalTimes = []
    finalVals = []
    finalValsSecondary = []
    for r in clusters:
        curr_cluster = triggerVals[r[0]:r[1]]
        allowed_indices = np.where(curr_cluster < highCutoff)[0]
        try:
            #If no point is below the upper threshold, this part will fail
            max_allowed_val = max(curr_cluster[allowed_indices])
            max_idx = np.where(curr_cluster == max_allowed_val)[0][0]
            finalTimes.append(triggerTimes[r[0] + max_idx])
            finalVals.append(triggerVals[r[0] + max_idx])
            finalValsSecondary.append(triggerValsSecondary[r[0] + max_idx])
        except:
            pass
    
    #Find if trigger corresponds to actual injection
    for trigIdx, trig in enumerate(finalTimes):
        tmp = np.abs(trig - injectionTimes)
        min_idx = np.argmin(tmp)
        if tmp[min_idx] <= symmetricInjectionWindow:
            truePos.append((injectionTimes[min_idx], trig, finalVals[trigIdx], finalValsSecondary[trigIdx]))
        else:
            falsePos.append((trig, injectionTimes[min_idx] - trig, finalVals[trigIdx], finalValsSecondary[trigIdx]))
    
    #Find missed injections
    missed_out = []
    foundInj = [pt[0] for pt in truePos]
    with h5py.File(get_collect_path(), 'r') as f:
        for inj in injectionTimes:
            highLevel, lowLevel = time_to_file_index(inj)
            if highLevel in allowed and lowLevel in allowed[highLevel]:
                if not inj in foundInj:
                    primary = f[str(highLevel)][str(lowLevel)][ts_data][:]
                    secondary = f[str(highLevel)][str(lowLevel)][secondary_data][:]
                    
                    #Calculate which index corresponds to the injection time
                    low = boundaries(highLevel, lowLevel)[0]
                    dt = 0.25
                    some_idx = int((inj - low) / dt)
                    if inj - (low + some_idx * dt) > (low + (some_idx + 1) * dt) - inj:
                        some_idx += 1
                    some_idx -= int(65.5 / dt)
                    
                    if mode.lower() == 'snr':
                        missed_out.append((inj, primary[some_idx], secondary[some_idx]))
                    else:
                        missed_out.append((inj, secondary[some_idx], primary[some_idx]))
    
    return (truePos, falsePos, missed_out)

def getCombinedTriggersAtThreshold(threshold_snr, threshold_p, mode='snr', symmetricInjectionWindow=3., highCutoff=np.inf, clusterTime=1.0):
    """Get only triggers, that are triggers in both outputs.
    
    Arguments
    ---------
    threshold_snr : float
        The threshold above which a value is considered a trigger in the
        SNR time series
    threshold_p : float
        The threshold above which a value is considered a trigger in the
        p-value time series
    mode : {optional, str}
        One of 'snr', 'p-score'. Which of the two trigger values is used
        to find the maximum and thus the time position.
    symmetricInjectionWindow : {optional, float}
        Time window around injection, in which we consider a trigger to
        belong to that injection
    highCutoff : {optional, float}
        Above which value triggers in the specific mode are ignored when
        finding the maximum value in a trigger interval
    clusterTime : {optional, float}
        Time two individual triggers are allowed to be separated by and
        still belong to the same cluster. If this value is smaller than
        0.25 all triggers are treated individually and no clustering
        takes place.
    
    Returns
    -------
    truePos : List
        List of triggers that where correctly identified. Each entry is
        a tuple containing (injection time, recovered time, SNR value at
        recovered time, p-score at recovered time)
    falsePos : List
        List of tirggers that do not belong to an injection. Each entry
        is a tuple containing (recovered time, distance to nearest
        injection, SNR value at recovered time, p-score at recovered
        time)
    missed:
        List of injection times that were not found
    """
    truePos = []
    falsePos = []
    missed = []
    with h5py.File(os.path.join(get_results_path(), 'allowed_stats.hf5'), 'r') as f:
        injectionTimes = f['times'][:]
    
    allowed = availableData()
    
    snrTriggerTimes = []
    snrTriggerVals = []
    with h5py.File(get_collect_path(), 'r') as f:
        for highLevel in sorted(allowed.keys()):
            for lowLevel in sorted(allowed[highLevel]):
                snr = f[str(highLevel)][str(lowLevel)]['TimeSeries/snrTimeSeries/data'][:]
                times = f[str(highLevel)][str(lowLevel)]['TimeSeries/snrTimeSeries/sample_times'][:]
                times = boundaries(highLevel, lowLevel)[0] + 65.5 + times
                #Do I want the high cutoff here or do I veto it later?
                triggerIdx = np.where(snr > threshold_snr)[0]
                snrTriggerTimes = snrTriggerTimes + list(times[triggerIdx])
                snrTriggerVals = snrTriggerVals + list(snr[triggerIdx])
    
    snrTriggerTimes = np.array(snrTriggerTimes)
    snrTriggerVals = np.array(snrTriggerVals)
    
    #print("snrTriggerTimes.shape: {}".format(snrTriggerTimes.shape))
    
    boolTriggerTimes = []
    boolTriggerVals = []
    with h5py.File(get_collect_path(), 'r') as f:
        for highLevel in sorted(allowed.keys()):
            for lowLevel in sorted(allowed[highLevel]):
                bools = f[str(highLevel)][str(lowLevel)]['TimeSeries/p-valueTimeSeries/data'][:]
                times = f[str(highLevel)][str(lowLevel)]['TimeSeries/p-valueTimeSeries/sample_times'][:]
                times = boundaries(highLevel, lowLevel)[0] + 65.5 + times
                #Do I want the high cutoff here or do I veto it later?
                triggerIdx = np.where(bools > threshold_p)[0]
                boolTriggerTimes = boolTriggerTimes + list(times[triggerIdx])
                boolTriggerVals = boolTriggerVals + list(bools[triggerIdx])
    
    boolTriggerTimes = np.array(boolTriggerTimes)
    boolTriggerVals = np.array(boolTriggerVals)
    
    #print("boolTriggerTimes.shape: {}".format(boolTriggerTimes.shape))
    
    triggerTimes = []
    triggerSNR = []
    triggerBool = []
    
    for i, snrT in enumerate(snrTriggerTimes):
        idx = np.where(boolTriggerTimes == snrT)[0]
        if len(idx) > 0:
            triggerTimes.append(snrT)
            triggerSNR.append(snrTriggerVals[i])
            triggerBool.append(boolTriggerVals[idx[0]])
    
    triggerTimes = np.array(triggerTimes)
    triggerSNR = np.array(triggerSNR)
    triggerBool = np.array(triggerBool)
    
    #print(triggerTimes.shape)
    
    #Cluster the triggers, use the symmetricInjectionWindow as a time
    #during which the triggers belong to the same event
    clusters = clusterTriggerTimes(triggerTimes, time_span=clusterTime)
    
    #Find the actual trigger times and veto thos with too high values
    finalTimes = []
    finalSNR = []
    finalBool = []
    for r in clusters:
        if mode.lower() == 'snr':
            curr_cluster = triggerSNR[r[0]:r[1]]
        elif mode.lower() == 'p-score':
            curr_cluster = triggerBool[r[0]:r[1]]
        allowed_indices = np.where(curr_cluster < highCutoff)[0]
        try:
            #If no point is below the upper threshold, this part will fail
            max_allowed_val = max(curr_cluster[allowed_indices])
            max_idx = np.where(curr_cluster == max_allowed_val)[0][0]
            finalTimes.append(triggerTimes[r[0] + max_idx])
            finalSNR.append(triggerSNR[r[0] + max_idx])
            finalBool.append(triggerBool[r[0] + max_idx])
        except:
            pass
    
    #Find if trigger corresponds to actual injection
    for trigIdx, trig in enumerate(finalTimes):
        tmp = np.abs(trig - injectionTimes)
        min_idx = np.argmin(tmp)
        if tmp[min_idx] <= symmetricInjectionWindow:
            truePos.append((injectionTimes[min_idx], trig, finalSNR[trigIdx], finalBool[trigIdx]))
        else:
            falsePos.append((trig, injectionTimes[min_idx] - trig, finalSNR[trigIdx], finalBool[trigIdx]))
    
    #Find missed injections
    foundInj = [pt[0] for pt in truePos]
    for inj in injectionTimes:
        if not inj in foundInj:
            missed.append(inj)
    
    return (truePos, falsePos, missed)

def writeStepToFile(g, x, mode):
    truePos, falsePos, missed = getTriggersAtThreshold(x, mode=mode)
    if mode.lower() == 'snr':
        modeName = g.create_group('SNR')
    elif mode.lower() == 'p-score':
        modeName = g.create_group('p-score')
    modeTrue = modeName.create_group('TruePositives')
    modeFalse = modeName.create_group('FalsePositives')
    modeMissed = modeName.create_group('MissedStats')
    
    modeName.create_dataset('Missed', data=np.array([pt[0] for pt in missed]))
    modeMissed.create_dataset('InjectionTime', data=np.array([pt[0] for pt in missed]))
    modeMissed.create_dataset('RecoveredSNR', data=np.array([pt[1] for pt in missed]))
    modeMissed.create_dataset('RecoveredPScore', data=np.array([pt[2] for pt in missed]))
    modeTrue.create_dataset('InjectionTime', data=np.array([pt[0] for pt in truePos]))
    modeTrue.create_dataset('RecoveredTime', data=np.array([pt[1] for pt in truePos]))
    modeTrue.create_dataset('Value', data=np.array([pt[2] for pt in truePos]))
    modeTrue.create_dataset('SecondaryValue', data=np.array([pt[3] for pt in truePos]))
    modeFalse.create_dataset('RecoveredTime', data=np.array([pt[0] for pt in falsePos]))
    modeFalse.create_dataset('MinimalDistance', data=np.array([pt[1] for pt in falsePos]))
    modeFalse.create_dataset('Value', data=np.array([pt[2] for pt in falsePos]))
    modeFalse.create_dataset('SecondaryValue', data=np.array([pt[3] for pt in falsePos]))

def calculateStepwise(steps, fileName, mode):
    file_path = os.path.join(get_results_path(), fileName)
    bar = progress_tracker(len(steps), name='Calculating')
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('x', data=steps)
        for x in steps:
            x_group = f.create_group(str(x))
            writeStepToFile(x_group, x, mode)
            bar.iterate()
    return

def getFalseAlarmRate(stepFile, mode='snr', plot=True):
    stepFile = os.path.join(get_results_path(), stepFile)
    if mode.lower() == 'snr':
        group_name = 'SNR'
    elif mode.lower() == 'p-score':
        group_name = 'p-score'
    else:
        raise NotImplementedError('No mode called {}'.format(mode))
    
    with h5py.File(stepFile, 'r') as f:
        x = f['x'][:]
        y = np.array([len(f[str(i)][group_name]['FalsePositives/RecoveredTime']) for i in x])
    
    available = availableData()
    
    num_available = sum([len(available[k]) for k in available.keys()])
    
    observation_time = num_available * 4096.
    
    seconds_per_month = 60 * 60 * 24 * 30
    
    y = y / observation_time * seconds_per_month
    
    #print(seconds_per_month / observation_time)
    
    false_alarm_file = os.path.splitext(stepFile)[0] + '_false_alarm.hf5'
    
    with h5py.File(false_alarm_file, 'w') as f:
        f.create_dataset('x', data=x)
        f.create_dataset('y', data=y)
    
    if plot:
        import matplotlib.pyplot as plt
        plt.semilogy(x, y)
        plt.grid()
        plt.savefig(os.path.splitext(false_alarm_file)[0] + '.png')
        plt.show()

def getSensitivity(stepFile, falseAlarmFile=None, mode='snr', plot=True):
    stepFile = os.path.join(get_results_path(), stepFile)
    if falseAlarmFile == None:
        falseAlarmFile = os.path.splitext(stepFile)[0] + '_false_alarm.hf5'
    else:
        falseAlarmFile = os.path.join(get_results_path(), falseAlarmFile)
    if mode.lower() == 'snr':
        group_name = 'SNR'
    elif mode.lower() == 'p-score':
        group_name = 'p-score'
    else:
        raise NotImplementedError('No mode called {}'.format(mode))
    
    with h5py.File(falseAlarmFile, 'r') as f:
        fa = f['y'][:]
    
    with h5py.File(stepFile, 'r') as f:
        x = f['x'][:]
        injTimesFound = [f[str(i)][group_name]['TruePositives/InjectionTime'][:] for i in x]
        injTimesMissed = [f[str(i)][group_name]['Missed'][:] for i in x]
    
    with h5py.File(get_stats_path(), 'r') as f:
        allInjTimes = f['times'][:]
        allInjDist = f['dist'][:]
        allInjM1 = f['mass1'][:]
        allInjM2 = f['mass2'][:]
    
    injDistFound = []
    injM1Found = []
    injM2Found = []
    injDistMissed = []
    injM1Missed = []
    injM2Missed = []
    totalIdx = np.arange(len(allInjTimes), dtype=int)
    for injFound, injMissed in zip(injTimesFound, injTimesMissed):
        tmpIdx = []
        for inj in injFound:
            tmpIdx.append(np.where(allInjTimes == inj)[0][0])
        if len(tmpIdx) > 0:
            tmpIdx = np.array(list(set(tmpIdx)), dtype=int)
            injDistFound.append(allInjDist[tmpIdx])
            injM1Found.append(allInjM1[tmpIdx])
            injM2Found.append(allInjM2[tmpIdx])
        else:
            injDistFound.append(None)
            injM1Found.append([1])
            injM2Found.append([1])
        missedIdx = []
        for inj in injMissed:
            missedIdx.append(np.where(allInjTimes == inj)[0][0])
        missedIdx = np.array(list(set(missedIdx)), dtype=int)
        injDistMissed.append(allInjDist[missedIdx])
        injM1Missed.append(allInjM1[missedIdx])
        injM2Missed.append(allInjM2[missedIdx])
    
    injMchirpFound = []
    for i in range(len(injM1Found)):
        tmpMchirp = []
        for j in range(len(injM1Found[i])):
            tmpMchirp.append(mchirp(injM1Found[i][j], injM2Found[i][j]))
        injMchirpFound.append(tmpMchirp)
    
    injMchirpMissed = []
    for i in range(len(injM1Missed)):
        tmpMchirp = []
        for j in range(len(injM1Missed[i])):
            tmpMchirp.append(mchirp(injM1Missed[i][j], injM2Missed[i][j]))
        injMchirpMissed.append(tmpMchirp)
    
    print("Starting to calculate volume")
    
    y = []
    bar = progress_tracker(len(injDistFound), name='Calculating volume')
    for i in range(len(injDistFound)):
        #print(type(injDistFound[i]))
        if isinstance(injDistFound[i], type(np.array([]))):
            y.append(volume_montecarlo(np.array(injDistFound[i]), np.array(injDistMissed[i]), np.array(injMchirpFound[i]), np.array(injMchirpMissed[i]), 'distance', 'volume', 'distance'))
        else:
            y.append((0, 0))
        bar.iterate()
    
    err = [pt[1] for pt in y]
    err_rad = [(3 * pt / (4 * np.pi))**(1. / 3.) for pt in err]
    vol = [pt[0] for pt in y]
    rad = [(3 * pt / (4 * np.pi))**(1. / 3.) for pt in vol]
    per = [float(len(injTimesFound[i]))/float(len(injTimesFound[i]) + len(injTimesMissed[i])) for i in range(len(injTimesMissed))]
    
    sensitivity_file = os.path.splitext(stepFile)[0] + '_sensitivty.hf5'
    
    with h5py.File(sensitivity_file, 'w') as f:
        f.create_dataset('x', data=fa)
        f.create_dataset('error_volume', data=np.array(err))
        f.create_dataset('error_radius', data=np.array(err_rad))
        f.create_dataset('volume', data=np.array(vol))
        f.create_dataset('radius', data=np.array(rad))
        f.create_dataset('percentage', data=np.array(per))
    
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.semilogx(fa, np.array(rad))
        ax.grid()
        ax.set_xlabel('False alarms per 30 days')
        ax.set_ylabel('Radius of sensitive sphere in MPc')
        x_low, x_high = ax.get_xlim()
        ax.set_xlim(x_high, x_low)
        plt.savefig(os.path.splitext(sensitivity_file)[0] + '.png')
        plt.show()

def writeCombinedStepToFile(g, x, y, mode):
    truePos, falsePos, missed = getCombinedTriggersAtThreshold(x, y, mode=mode)
    modeTrue = g.create_group('TruePositives')
    modeFalse = g.create_group('FalsePositives')
    
    g.create_dataset('Missed', data=np.array(missed))
    
    modeTrue.create_dataset('InjectionTime', data=np.array([pt[0] for pt in truePos]))
    modeTrue.create_dataset('RecoveredTime', data=np.array([pt[1] for pt in truePos]))
    modeTrue.create_dataset('RecoveredSNR', data=np.array([pt[2] for pt in truePos]))
    modeTrue.create_dataset('RecoveredP-score', data=np.array([pt[3] for pt in truePos]))
    
    modeFalse.create_dataset('RecoveredTime', data=np.array([pt[0] for pt in falsePos]))                                                        
    modeFalse.create_dataset('MinimalDistance', data=np.array([pt[1] for pt in falsePos]))
    modeFalse.create_dataset('RecoveredSNR', data=np.array([pt[2] for pt in falsePos]))
    modeFalse.create_dataset('RecoveredP-score', data=np.array([pt[3] for pt in falsePos]))
    return

def calculateStepwiseCombined(snrSteps, pSteps, fileName, mode):
    file_path = os.path.join(get_results_path(), fileName)
    bar = progress_tracker(len(snrSteps) * len(pSteps), name='Calculating steps')
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('snr', data=snrSteps)
        f.create_dataset('p-score', data=pSteps)
        f.create_dataset('mode', data=np.str(mode))
        for x in snrSteps:
            x_group = f.create_group(str(x))
            for y in pSteps:
                y_group = x_group.create_group(str(y))
                writeCombinedStepToFile(y_group, x, y, mode)
                bar.iterate()
    return

def getPureTriggers(threshold, mode='snr'):
    if mode.lower() == 'snr':
        ts_data = 'TimeSeries/snrTimeSeries/data'
        ts_times = 'TimeSeries/snrTimeSeries/sample_times'
    else:
        ts_data = 'TimeSeries/p-valueTimeSeries/data'
        ts_times = 'TimeSeries/p-valueTimeSeries/sample_times'
    allowed = availableData()
    triggerTimes = []
    triggerValues = []
    with h5py.File(get_collect_path(), 'r') as f:
        for highLevel in sorted(allowed.keys()):
            for lowLevel in sorted(allowed[highLevel]):
                data = f[str(highLevel)][str(lowLevel)][ts_data][:]
                times = f[str(highLevel)][str(lowLevel)][ts_times][:]
                times = boundaries(highLevel, lowLevel)[0] + 65.5 + times
                
                triggerIdx = np.where(data > threshold)[0]
                triggerTimes = triggerTimes + list(times[triggerIdx])
                triggerValues = triggerValues + list(data[triggerIdx])
    return np.array(triggerTimes), np.array(triggerValues)

def matchTriggers(snrTimes, pTimes, snrVals, pVals):
    if len(pTimes) < len(snrTimes):
        smallTimes = pTimes
        smallVals = pVals
        bigTimes = snrTimes
        bigVals = snrVals
    else:
        smallTimes = snrTimes
        smallVals = snrVals
        bigTimes = pTimes
        bigVals = pVals
    
    combinedTimes = []
    combinedSmallVals = []
    combinedBigVals = []
    for i, t in enumerate(smallTimes):
        idx = np.where(bigTimes == t)[0]
        if len(idx) > 0:
            combinedTimes.append(t)
            combinedSmallVals.append(smallVals[i])
            combinedBigVals.append(bigVals[idx[0]])
    
    if len(pTimes) < len(snrTimes):
        return np.array(combinedTimes), np.array(combinedBigVals), np.array(combinedSmallVals)
    else:
        return np.array(combinedTimes), np.array(combinedSmallVals), np.array(combinedBigVals)

def findCombinedInjections(snrTrigs, pTrigs, mode='snr', clusterTime=1., highCutoff=np.inf, symmetricInjectionWindow=3.):
    with h5py.File(os.path.join(get_results_path(), 'allowed_stats.hf5'), 'r') as f:
        injectionTimes = f['times'][:]
    
    trigTimes, snrVals, pVals = matchTriggers(snrTrigs[0], pTrigs[0], snrTrigs[1], pTrigs[1])
    clusters = clusterTriggerTimes(trigTimes, time_span=clusterTime)
    
    #Generate main triggers
    finalTimes = []
    finalSnr = []
    finalP = []
    for r in clusters:
        if mode.lower() == 'snr':
            curr_cluster = snrVals[r[0]:r[1]]
        elif mode.lower() == 'p-score':
            curr_cluster = pVals[r[0]:r[1]]
        else:
            raise NotImplementedError('Mode {} is not available'.format(mode))
        
        allowed_indices = np.where(curr_cluster < highCutoff)
        try:
            max_allowed_val = max(curr_cluster[allowed_indices])
            max_idx = np.where(curr_cluster == max_allowed_val)[0][0]
            finalTimes.append(trigTimes[r[0] + max_idx])
            finalSnr.append(snrVals[r[0] + max_idx])
            finalP.append(pVals[r[0] + max_idx])
        except:
            pass
    
    truePos = []
    falsePos = []
    missed = []
    
    for trigIdx, trig in enumerate(finalTimes):
        tmp = np.abs(trig - injectionTimes)
        min_idx = np.argmin(tmp)
        if tmp[min_idx] <= symmetricInjectionWindow:
            truePos.append((injectionTimes[min_idx], trig, finalSnr[trigIdx], finalP[trigIdx]))
        else:
            falsePos.append((trig, injectionTimes[min_idx] - trig, finalSnr[trigIdx], finalP[trigIdx]))
    
    foundInj = [pt[0] for pt in truePos]
    for inj in injectionTimes:
        if not inj in foundInj:
            missed.append(inj)
    return truePos, falsePos, missed

def writeCombinedResultsToFile(g, truePos, falsePos, missed):
    modeTrue = g.create_group('TruePositives')
    modeFalse = g.create_group('FalsePositives')
    
    g.create_dataset('Missed', data=np.array(missed))
    
    modeTrue.create_dataset('InjectionTime', data=np.array([pt[0] for pt in truePos]))
    modeTrue.create_dataset('RecoveredTime', data=np.array([pt[1] for pt in truePos]))
    modeTrue.create_dataset('RecoveredSNR', data=np.array([pt[2] for pt in truePos]))
    modeTrue.create_dataset('RecoveredP-score', data=np.array([pt[3] for pt in truePos]))
    
    modeFalse.create_dataset('RecoveredTime', data=np.array([pt[0] for pt in falsePos]))                                                        
    modeFalse.create_dataset('MinimalDistance', data=np.array([pt[1] for pt in falsePos]))
    modeFalse.create_dataset('RecoveredSNR', data=np.array([pt[2] for pt in falsePos]))
    modeFalse.create_dataset('RecoveredP-score', data=np.array([pt[3] for pt in falsePos]))
    return

def calculateStepwiseCombinedTest(snrSteps, pSteps, fileName, mode):
    file_path = os.path.join(get_results_path(), fileName)
    #Calculate triggers at threshold
    snrTrigs = {}
    bar = progress_tracker(len(snrSteps), name='Generating SNR triggers')
    for snr in snrSteps:
        snrTrigs[snr] = getPureTriggers(snr, mode='snr')
        bar.iterate()
    pTrigs = {}
    bar = progress_tracker(len(pSteps), name='Generating p-score triggers')
    for p in pSteps:
        pTrigs[p] = getPureTriggers(p, mode='p-score')
        bar.iterate()
    
    bar = progress_tracker(len(snrSteps) * len(pSteps), name='Calculating steps')
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('snr', data=snrSteps)
        f.create_dataset('p-score', data=pSteps)
        f.create_dataset('mode', data=np.str(mode))
        for x in snrSteps:
            x_group = f.create_group(str(x))
            for y in pSteps:
                y_group = x_group.create_group(str(y))
                truePos, falsePos, missed = findCombinedInjections(snrTrigs[x], pTrigs[y], mode=mode)
                writeCombinedResultsToFile(y_group, truePos, falsePos, missed)
                bar.iterate()
    return

def getCombinedFalseAlarm(stepFile, mode='snr', plot=True):
    stepFile = os.path.join(get_results_path(), stepFile)
    
    with h5py.File(stepFile, 'r') as f:
        snr = f['snr'][:]
        p = f['p-score'][:]
        x, y = np.meshgrid(snr, p)
        x = x.flatten()
        y = y.flatten()
        z = np.array([len(f[str(x[i])][str(y[i])]['FalsePositives/RecoveredTime']) for i in range(len(x))])
    
    available = availableData()
    
    num_available = sum([len(available[k]) for k in available.keys()])
    
    observation_time = num_available * 4096.
    
    seconds_per_month = 60 * 60 * 24 * 30
    
    z = z / observation_time * seconds_per_month
    
    false_alarm_file = os.path.splitext(stepFile)[0] + '_false_alarm.hf5'
    
    with h5py.File(false_alarm_file, 'w') as f:
        f.create_dataset('x', data=x)
        f.create_dataset('y', data=y)
        f.create_dataset('z', data=z)
        f.create_dataset('mode', data=np.str(mode))
    
    if plot:
        import matplotlib.pyplot as plt
        from mpl_toolkits import mplot3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        shape = (int(np.sqrt(len(x))), int(np.sqrt(len(x))))
        ax.plot_wireframe(x.reshape(shape), y.reshape(shape), np.log10(z.reshape(shape)))
        ax.set_xlabel('SNR threshold')
        ax.set_ylabel('p-score threshold')
        ax.set_zlabel('False alarms per month in 10^z')
        plt.savefig(os.path.splitext(false_alarm_file)[0] + '.png')
        plt.show()
    return

def getCombinedSensitivity(stepFile, falseAlarmFile=None, mode='snr', plot=None):
    stepFile = os.path.join(get_results_path(), stepFile)
    if falseAlarmFile == None:
        falseAlarmFile = os.path.splitext(stepFile)[0] + '_false_alarm.hf5'
    else:
        falseAlarmFile = os.path.join(get_results_path(), falseAlarmFile)
    
    if mode.lower() == 'snr':
        group_name = 'SNR'
    elif mode.lower() == 'p-score':
        group_name = 'p-score'
    else:
        raise NotImplementedError('No mode called {}'.format(mode))
    
    #Load necessary data
    with h5py.File(falseAlarmFile, 'r') as f:
        fa = f['z'][:]
    
    with h5py.File(stepFile, 'r') as f:
        snr = f['snr'][:]
        p = f['p-score'][:]
        injTimesFound = {}
        injTimesMissed = {}
        for x in snr:
            injTimesFound[x] = {y: f[str(x)][str(y)]['TruePositives/InjectionTime'][:] for y in p}
            injTimesMissed[x] = {y: f[str(x)][str(y)]['Missed'][:] for y in p}
    
    with h5py.File(get_stats_path(), 'r') as f:
        allInjTimes = f['times'][:]
        allInjDist = f['dist'][:]
        allInjM1 = f['mass1'][:]
        allInjM2 = f['mass2'][:]
    
    injDistFound = {x: {} for x in snr}
    injM1Found = {x: {} for x in snr}
    injM2Found = {x: {} for x in snr}
    injMchirpFound = {x: {} for x in snr}
    injDistMissed = {x: {} for x in snr}
    injM1Missed = {x: {} for x in snr}
    injM2Missed = {x: {} for x in snr}
    injMchirpMissed = {x: {} for x in snr}
    totalIdx = np.arange(len(allInjTimes), dtype=int)
    bar = progress_tracker(len(snr) * len(p), name='Getting found and missed stats')
    for x in snr:
        for y in p:
            tmpIdx = []
            for inj in injTimesFound[x][y]:
                tmpIdx.append(np.where(allInjTimes == inj)[0][0])
            if len(tmpIdx) > 0:
                tmpIdx = np.array(list(set(tmpIdx)), dtype=int)
                injDistFound[x][y] = allInjDist[tmpIdx]
                injM1Found[x][y] = allInjM1[tmpIdx]
                injM2Found[x][y] = allInjM2[tmpIdx]
                injMchirpFound[x][y] = np.array([mchirp(injM1Found[x][y][i], injM2Found[x][y][i]) for i in range(len(injM1Found[x][y]))])
            else:
                injDistFound[x][y] = None
                injM1Found[x][y] = [1]
                injM2Found[x][y] = [1]
            
            missedIdx = np.setdiff1d(totalIdx, tmpIdx)
            injDistMissed[x][y] = allInjDist[missedIdx]
            injM1Missed[x][y] = allInjM1[missedIdx]
            injM2Missed[x][y] = allInjM2[missedIdx]
            injMchirpMissed[x][y] = np.array([mchirp(injM1Missed[x][y][i], injM2Missed[x][y][i]) for i in range(len(injM1Missed[x][y]))])
            
            bar.iterate()
    
    print("Starting to calculate volume")
    
    z = [[] for x in snr]
    bar = progress_tracker(len(snr) * len(p), name='Calculating volume')
    for i, x in enumerate(snr):
        for j, y in enumerate(p):
            if isinstance(injDistFound[x][y], type(np.array([]))):
                z[i].append(volume_montecarlo(injDistFound[x][y], injDistMissed[x][y], injMchirpFound[x][y], injMchirpMissed[x][y], 'distance', 'volume', 'distance'))
            else:
                z[i].append((0,0))
            bar.iterate()
    
    err = [[pt[1] for pt in l] for l in z]
    err_rad = [[(3 * pt[1] / (4 * np.pi))**(1. / 3.) for pt in l] for l in z]
    vol = err = [[pt[0] for pt in l] for l in z]
    rad = [[(3 * pt[0] / (4 * np.pi))**(1. / 3.) for pt in l] for l in z]
    per = [[float(len(injTimesFound[x][y])) / float(len(allInjTimes)) for y in p] for x in snr]
    
    sens_file = os.path.splitext(stepFile)[0] + '_sensitivity.hf5'
    
    X, Y = np.meshgrid(snr, p)
    
    with h5py.File(sens_file, 'w') as f:
        f.create_dataset('snr', data=X.flatten())
        f.create_dataset('p-score', data=Y.flatten())
        f.create_dataset('xy', data=fa)
        f.create_dataset('error_volume', data=np.array(err).flatten())
        f.create_dataset('error_radius', data=np.array(err_rad).flatten())
        f.create_dataset('volume', data=np.array(vol).flatten())
        f.create_dataset('radius', data=np.array(rad).flatten())
        f.create_dataset('percentage', data=np.array(per).flatten())
    
    Z = np.array(rad)
    
    
    if not plot == None:
        if mode.lower() == 'snr':
            minIdx = np.argmin(np.abs(p - plot))
            idx = np.where(Y == p[minIdx])
            tit = 'p-score = {}'.format(p[minIdx])
        elif mode.lower() == 'p-score':
            minIdx = np.argmin(np.abs(snr - plot))
            idx = np.where(Y == snr[minIdx])
            tit = 'SNR = {}'.format(snr[minIdx])
        else:
            raise NotImplementedError('No mode called {}'.format(mode))
        
        FA = fa.reshape(X.shape)
        x = [FA[idx[0][i]][idx[1][i]] for i in range(len(idx[0]))]
        y = [Z[idx[0][i]][idx[1][i]] for i in range(len(idx[0]))]
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.semilogx(x, y)
        ax.grid()
        ax.set_xlabel('False alarms per 30 days')
        ax.set_ylabel('Radius of sensitive sphere in MPc')
        ax.set_title(tit)
        x_low, x_high = ax.get_xlim()
        ax.set_xlim(x_high, x_low)
        plt.savefig(os.path.splitext(sens_file)[0] + '.png')
        plt.show()

def getSnrs():
    from pycbc.waveform import get_td_waveform
    from pycbc.detector import Detector
    from pycbc.psd import aLIGOZeroDetHighPower
    from pycbc.filter import sigma
    filePath = get_stats_path()
    newFilePath = os.path.splitext(filePath)[0] + '_snr.hf5'
    L1 = Detector('L1')
    H1 = Detector('H1')
    with h5py.File(filePath, 'r') as old:
        i = 0
        dist = old['dist'][i]
        mass1 = old['mass1'][i]
        mass2 = old['mass2'][i]
        dec = old['dec'][i]
        inc = old['inc'][i]
        cphase = old['cphase'][i]
        pol = old['pol'][i]
        ra = old['ra'][i]
        hp, hc = get_td_waveform(approximant='TaylorF2', f_lower=20.0, delta_t=1.0/4096, mass1=mass1, mass2=mass2, distance=dist, coa_phase=cphase, inclination=inc)
        l1 = L1.project_wave(hp, hc, ra, dec, pol)
        h1 = H1.project_wave(hp, hc, ra, dec, pol)
        psd = aLIGOZeroDetHighPower(length=len(h1.to_frequencyseries()), delta_f=h1.delta_f, low_freq_cutoff=20.0)
        with h5py.File(newFilePath, 'w') as new:
            for k in old.keys():
                new.create_dataset(k, data=old[k][:])
            snrs = []
            bar = progress_tracker(len(old['dist']), name='Calculating SNRs')
            for i in range(len(old['dist'])):
                dist = old['dist'][i]
                mass1 = old['mass1'][i]
                mass2 = old['mass2'][i]
                dec = old['dec'][i]
                inc = old['inc'][i]
                cphase = old['cphase'][i]
                pol = old['pol'][i]
                ra = old['ra'][i]
                hp, hc = get_td_waveform(approximant='TaylorF2', f_lower=20.0, delta_t=1.0/4096, mass1=mass1, mass2=mass2, distance=dist, coa_phase=cphase, inclination=inc)
                l1 = L1.project_wave(hp, hc, ra, dec, pol)
                h1 = H1.project_wave(hp, hc, ra, dec, pol)
                #print("L1 length: {}".format(len(l1.to_frequencyseries())))
                #print("L1 delta_f: {}".format(l1.delta_f))
                #print("H1 length: {}".format(len(h1.to_frequencyseries())))
                #print("H1 delta_f: {}".format(h1.delta_f))
                #psdl1 = aLIGOZeroDetHighPower(length=len(l1.to_frequencyseries()), delta_f=l1.delta_f, low_freq_cutoff=20.0)
                #psdh1 = aLIGOZeroDetHighPower(length=len(h1.to_frequencyseries()), delta_f=h1.delta_f, low_freq_cutoff=20.0)
                l1snr = sigma(l1, psd=psd, low_frequency_cutoff=20.0)
                h1snr = sigma(h1, psd=psd, low_frequency_cutoff=20.0)
                snrs.append(np.sqrt(l1snr**2 + h1snr**2))
                bar.iterate()
            new.create_dataset('snr', data=np.array(snrs))
    return

def getSensitivitySnr(stepFile, falseAlarmRate, falseAlarmFile=None, mode='snr', plot=True):
    stepFile = os.path.join(get_results_path(), stepFile)
    if falseAlarmFile == None:
        falseAlarmFile = os.path.splitext(stepFile)[0] + '_false_alarm.hf5'
    else:
        falseAlarmFile = os.path.join(get_results_path(), falseAlarmFile)
    if mode.lower() == 'snr':
        group_name = 'SNR'
    elif mode.lower() == 'p-score':
        group_name = 'p-score'
    else:
        raise NotImplementedError('No mode called {}'.format(mode))
    
    with h5py.File(falseAlarmFile, 'r') as f:
        fa = f['y'][:]
        x = f['x'][:]
    
    idx = np.argmin(np.abs(fa - falseAlarmRate))
    x = x[idx]
    fa = fa[idx]
    
    with h5py.File(stepFile, 'r') as f:
        injTimesFound = f[str(x)][group_name]['TruePositives/InjectionTime'][:]
        injTimesMissed = f[str(x)][group_name]['Missed'][:]
    
    with h5py.File(get_snr_stats_path(), 'r') as f:
        allInjTimes = f['times'][:]
        allInjSnr = f['snr'][:]
    
    allowed = availableData()
    
    tmpTimes = []
    tmpSnrs = []
    for i in range(len(allInjTimes)):
        highLevel, lowLevel = time_to_file_index(allInjTimes[i])
        if highLevel in allowed and lowLevel in allowed[highLevel]:
            tmpTimes.append(allInjTimes[i])
            tmpSnrs.append(allInjSnr[i])
    allInjTimes = np.array(tmpTimes)
    allInjSnr = np.array(tmpSnrs)
    
    tmpIdx = []
    for inj in injTimesFound:
        tmpIdx.append(np.where(allInjTimes == inj)[0][0])
    if len(tmpIdx) > 0:
        tmpIdx = np.array(list(set(tmpIdx)), dtype=int)
        injSnrFound = allInjSnr[tmpIdx]
    else:
        injSnrFound = None
    
    bins = np.arange(int(np.floor(min(allInjSnr))), int(np.ceil(max(allInjSnr))), 1)
    
    maxIdx = 56 - min(bins) + 1
    
    binIdxFound = np.digitize(injSnrFound, bins)
    binIdxNormal = np.digitize(allInjSnr, bins)
    
    numFound = np.bincount(binIdxFound, minlength=len(bins))
    numTotal = np.bincount(binIdxNormal, minlength=len(bins))
    
    y = []
    for i in range(len(numFound)):
        if numTotal[i] == 0:
            y.append(1)
        else:
            y.append(float(numFound[i]) / float(numTotal[i]))
    
    y = np.array(y)
    
    sensitivity_file = os.path.splitext(stepFile)[0] + '_sensitivty_snr.hf5'
    
    with h5py.File(sensitivity_file, 'w') as f:
        f.create_dataset('bins', data=bins)
        f.create_dataset('false_alarm_rate', data=np.float(fa))
        f.create_dataset('y', data=np.array(y))
    
    print("Used false alarm rate: {}".format(fa))
    
    if plot:
        #Only plot to SNR 56 -> pSNR = 1
        maxIdx = 56 - min(bins) + 1
        plot_bins = np.arange(min(bins)-0.5, max(bins)+0.6, 1)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        #ax.bar(plot_bins[:maxIdx] / (39.225 * np.sqrt(2)), y[:maxIdx], width=1 / (39.225 * np.sqrt(2)))
        ax.plot(plot_bins[:maxIdx] / (39.225 * np.sqrt(2)), y[:maxIdx])
        ax.grid()
        ax.set_xlabel('pSNR')
        ax.set_ylabel('Ratio of found signals')
        plt.savefig(os.path.splitext(sensitivity_file)[0] + '.png')
        plt.show()
    
    return numFound, numTotal

#def getTriggersAtThresholdCluster(threshold, mode='snr', symmetricInjectionWindow=0.25, highCutoff=np.inf, clusterTime=1.0):
def getTriggersAtThresholdCluster(threshold, mode='snr', symmetricInjectionWindow=0.25, highCutoff=50., clusterTime=1.0):
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
    clusterTime : {optional, float}
        Time two individual triggers are allowed to be separated by and
        still belong to the same cluster. If this value is smaller than
        0.25 all triggers are treated individually and no clustering
        takes place.
    
    Returns
    -------
    truePos : List
        List of triggers that where correctly identified. Each entry is
        a tuple containing (injection time, recovered time, recovered
        value at that time)
    falsePos : List
        List of tirggers that do not belong to an injection. Each entry
        is a tuple containing (recovered time, distance to nearest
        injection, value at recovered time)
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
    clusters = clusterTriggerTimes(triggerTimes, time_span=clusterTime)
    
    #Find the actual trigger times and veto thos with too high values
    finalTimes = []
    finalVals = []
    for r in clusters:
        curr_cluster = triggerVals[r[0]:r[1]]
        allowed_idx = np.where(curr_cluster < highCutoff)[0]
        try:
            #If no point is below the upper threshold, this part will fail
            min_idx = np.min(allowed_idx)
            max_idx = np.max(allowed_idx)
            if max_idx - min_idx > 5:
                finalTimes.append((triggerTimes[r[0] + min_idx], triggerTimes[r[0] + max_idx]))
                finalVals.append([triggerVals[r[0] + pt] for pt in np.arange(min_idx, max_idx+1, 1, dtype=int)])
        except:
            pass
    
    #Find if trigger corresponds to actual injection
    for trigIdx, trig in enumerate(finalTimes):
        big = injectionTimes < (trig[1] + symmetricInjectionWindow)
        small = injectionTimes > (trig[0] - symmetricInjectionWindow)
        contained_injections = np.where(np.logical_and(big, small))[0]
        if len(contained_injections) > 1:
            print("Found interval that contains {} injections.".format(len(contained_injections)))
        if len(contained_injections) < 1:
            min_idx = np.argmin(np.abs(injectionTimes - trig[0]))
            max_idx = np.argmin(np.abs(injectionTimes - trig[1]))
            falsePos.append((np.array(trig), np.array([injectionTimes[min_idx], injectionTimes[max_idx]]), np.array(finalVals[trigIdx])))
        else:
            truePos.append((injectionTimes[contained_injections[0]], np.array(trig), np.array(finalVals[trigIdx])))
    
    #Find missed injections
    foundInj = [pt[0] for pt in truePos]
    for inj in injectionTimes:
        if not inj in foundInj:
            missed.append(inj)
    
    return (truePos, falsePos, missed)

def writeStepToFileCluster(g, x, mode):
    truePos, falsePos, missed = getTriggersAtThresholdCluster(x, mode=mode)
    if mode.lower() == 'snr':
        modeName = g.create_group('SNR')
    elif mode.lower() == 'p-score':
        modeName = g.create_group('p-score')
    modeTrue = modeName.create_group('TruePositives')
    modeFalse = modeName.create_group('FalsePositives')
    
    #print(truePos[0][2])
    
    true_pos_vals_size = max([len(pt[2]) for pt in truePos])
    false_pos_vals_size = max([len(pt[2]) for pt in falsePos])
    
    modeName.create_dataset('Missed', data=np.array(missed))
    modeTrue.create_dataset('InjectionTime', data=np.array([pt[0] for pt in truePos]))
    modeTrue.create_dataset('RecoveredTime', data=np.array([pt[1] for pt in truePos]))
    modeTrue.create_dataset('Value', data=np.zeros((len(truePos), true_pos_vals_size)))
    for i, pt in enumerate(truePos):
        modeTrue['Value'][i][:len(pt[2])] = pt[2][:]
    modeFalse.create_dataset('RecoveredTime', data=np.array([pt[0] for pt in falsePos]))
    modeFalse.create_dataset('MinimalDistance', data=np.array([pt[1] for pt in falsePos]))
    modeFalse.create_dataset('Value', data=np.zeros((len(falsePos), false_pos_vals_size)))
    for i, pt in enumerate(falsePos):
        modeFalse['Value'][i][:len(pt[2])] = pt[2][:]

def calculateStepwiseCluster(steps, fileName, mode):
    file_path = os.path.join(get_results_path(), fileName)
    bar = progress_tracker(len(steps), name='Calculating')
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('x', data=steps)
        for x in steps:
            x_group = f.create_group(str(x))
            writeStepToFileCluster(x_group, x, mode)
            bar.iterate()
    return

def pScoreDistribution(low_limit, high_limit, mode='snr'):
    if mode.lower() == 'snr':
        ts_data = 'TimeSeries/snrTimeSeries/data'
        ts_times = 'TimeSeries/snrTimeSeries/sample_times'
    elif mode.lower() == 'p-score':
        ts_data = 'TimeSeries/p-valueTimeSeries/data'
        ts_times = 'TimeSeries/p-valueTimeSeries/sample_times'
    else:
        raise NotImplementedError('Unsupported type {}'.format(mode))
    
    allowed = availableData()
    
    above = 0
    below = 0
    total = 0
    with h5py.File(get_collect_path(), 'r') as f:
        for highLevel in sorted(allowed.keys()):
            for lowLevel in sorted(allowed[highLevel]):
                try:
                    total += len(f[str(highLevel)][str(lowLevel)][ts_data])
                    values = f[str(highLevel)][str(lowLevel)][ts_data][:]
                    above += (values > high_limit).sum()
                    below += (values < low_limit).sum()
                except KeyError:
                    t, v, tb = sys.exc_info()
                    print("Tried to load highLevel {} and lowLevel {}. Got a KeyError.".format(lowLevel, highLevel))
                    raise t, v, tb
    
    return below, above, total, float(below) / float(total), float(above) / float(total)

def binRawData(bins=10, mode='snr'):
    if mode.lower() == 'snr':
        ts_data = 'TimeSeries/snrTimeSeries/data'
        ts_times = 'TimeSeries/snrTimeSeries/sample_times'
    elif mode.lower() == 'p-score':
        ts_data = 'TimeSeries/p-valueTimeSeries/data'
        ts_times = 'TimeSeries/p-valueTimeSeries/sample_times'
    else:
        raise NotImplementedError('Unsupported type {}'.format(mode))
    
    allowed = availableData()
    
    data = None
    bin_edges = None
    with h5py.File(get_collect_path(), 'r') as f:
        for highLevel in sorted(allowed.keys()):
            for lowLevel in sorted(allowed[highLevel]):
                if data is None:
                    data, bin_edges = np.histogram(f[str(highLevel)][str(lowLevel)][ts_data][:], bins=bins)
                else:
                    data += np.histogram(f[str(highLevel)][str(lowLevel)][ts_data][:], bins=bins)[0]
    
    return data, bin_edges
