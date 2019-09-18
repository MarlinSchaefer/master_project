import h5py
import numpy as np
from aux_functions import get_store_path
import os
from progress_bar import progress_tracker
from pycbc.sensitivity import volume_montecarlo

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
    for trigIdx, trig in enumerate(finalTimes):
        tmp = np.abs(trig - injectionTimes)
        min_idx = np.argmin(tmp)
        if tmp[min_idx] <= symmetricInjectionWindow:
            truePos.append((tmp[min_idx], trig, finalVals[trigIdx]))
        else:
            falsePos.append((trig, injectionTimes[min_idx] - trig, finalVals[trigIdx]))
    
    #Find missed injections
    foundInj = [pt[0] for pt in truePos]
    for inj in injectionTimes:
        if not inj in foundInj:
            missed.append(inj)
    
    return (truePos, falsePos, missed)

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
    
    modeName.create_dataset('Missed', data=np.array(missed))
    modeTrue.create_dataset('InjectionTime', data=np.array([pt[0] for pt in truePos]))
    modeTrue.create_dataset('RecoveredTime', data=np.array([pt[1] for pt in truePos]))
    modeTrue.create_dataset('Value', data=np.array([pt[2] for pt in truePos]))
    modeFalse.create_dataset('RecoveredTime', data=np.array([pt[0] for pt in falsePos]))
    modeFalse.create_dataset('MinimalDistance', data=np.array([pt[1] for pt in falsePos]))
    modeFalse.create_dataset('Value', data=np.array([pt[2] for pt in falsePos]))

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

def getSensitivity(stepFile, falseAlarmFile, mode='snr', plot=True):
    stepFile = os.path.join(get_results_path(), stepFile)
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
        injTimesFound = [f[str(i)][group_name]['TruePositives/RecoveredTime'][:] for i in x]
        injTimesMissed = [f[str(i)][group_name]['Missed'][:] for i in x]
    
    with h5py.File(get_stats_path(), 'r') as f:
        allInjTimes = f['times'][:]
        allInjDist = f['dist'][:]
        allInjM1 = f['mass1'][:]
        allInjM2 = f['mass2'][:]
    
    injDistFound = []
    injM1Found = []
    injM2Found = []
    for injFound in injTimesFound:
        tmpDist = []
        tmpM1 = []
        tmpM2 = []
        for inj in injFound:
            idx = np.where(allInjTimes == inj)[0][0]
            tmpDist.append(allInjDist[idx])
            tmpM1.append(allInjM1[idx])
            tmpM2.append(allInjM2[idx])
        injDistFound.append(tmpDist)
        injM1Found.append(tmpM1)
        injM2Found.append(tmpM2)
    injDistMissed = []
    injM1Missed = []
    injM2Missed = []
    for injMissed in injTimesMissed:
        tmpDist = []
        tmpM1 = []
        tmpM2 = []
        for inj in injMissed:
            idx = np.where(allInjTimes == inj)[0][0]
            tmpDist.append(allInjDist[idx])
            tmpM1.append(allInjM1[idx])
            tmpM2.append(allInjM2[idx])
        injDistMissed.append(tmpDist)
        injM1Missed.append(tmpM1)
        injM2Missed.append(tmpM2)
    
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
    
    y = [volume_montecarlo(injDistFound[i], injDistMissed[i], injMchirpFound[i], injMchirpMissed[i], 'distance', 'volume', 'distance')]
    
    err = [pt[1] for pt in y]
    err_rad = [(3 * pt / (4 * np.pi))**(1. / 3.) for pt in err]
    vol = [pt[0] for pt in y]
    rad = [(3 * pt / (4 * np.pi))**(1. / 3.) for pt in vol]
    
    sensitivity_file = os.path.splitext(stepFile)[0] + '_sensitivty.hf5'
    
    with h5py.File(sensitivity_file, 'w') as f:
        f.create_dataset('x', data=fa)
        f.create_dataset('error_volume', data=np.array(err))
        f.create_dataset('error_radius', data=np.array(err_rad))
        f.create_dataset('volume', data=np.array(vol))
        f.create_dataset('radius', data=np.array(rad))
    
    if plot:
        import matplotlib.pyplot as plt
        plt.semilogx(np.flip(fa), np.array(rad))
        plt.grid()
        plt.savefig(os.path.splitext(sensitivity_file)[0] + '.png')
        plt.xlabel('False alarms per 30 days')
        plt.ylabel('Radius of sensitive sphere')
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
