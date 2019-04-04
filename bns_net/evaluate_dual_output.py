import numpy as np
import keras
from load_data import load_testing_data, load_testing_labels, load_testing_calculated_snr
from run_net import get_store_path
import os
import matplotlib.pyplot as plt
from make_snr_plot import _do_plot

def evaluate_dual_output(net_name, temp_name, show=True):
    """Evaluates the performance of a network with two outputs.
    
    This function expects the network to have two outputs. The first should be
    the predicted SNR while the second is a representation of a boolean value
    that indicates whether there is a GW in the data or not. This boolean value
    should be an array with two entries (p, 1-p), where p is the "probabilty"
    of a GW being present. Thus the structure of a single output needs to be:
    [SNR, [p, 1-p]]
    This function also creates a few plots.
    
    Arguments
    ---------
    net_name : str
        The name of the networks '.hf5' file (file extension NOT included).
    
    temp_name : str
        The name of the datas '.hf5' file (file extension NOT included).
    
    Returns
    -------
    list
        A list with five values. The first entry represents how many signals
        the network correctly predicted to have a GW in them. The second
        represents how many signals the network correctly predicted to have no
        GW in them. The third how many it falsly predicted to have no GW in the
        data, the fourth how many it falsly predicted to have a GW in the data,
        the fifth represents the number of samples where the network had a bias
        of less then 60% towards one or the other output. (i.e. the output for
        the bool value was something like [0.55, 0.45])
    """
    saves_path = get_store_path()
    net_path = os.path.join(saves_path, net_name + '.hf5')
    temp_path = os.path.join(saves_path, temp_name + '.hf5')
    
    net = keras.models.load_model(net_path)
    
    te_d = load_testing_data(temp_path)
    te_l = load_testing_labels(temp_path)
    te_c = load_testing_calculated_snr(temp_path)
    
    res = net.predict(te_d, verbose=1)
    
    predicted_snr = [pt[0] for pt in res[0]]
    
    predicted_bool = [pt[0] > pt[1] for pt in res[1]]
    
    l = [0, 0, 0, 0, 0]
    
    for i in range(len(predicted_bool)):
        if predicted_bool[i] == bool(te_l[i][1]):
            #Correctly predicted
            if predicted_bool[i]:
                #GW is in the signal
                l[0] += 1
            else:
                #GW is not in the signal
                l[1] += 1
        else:
            #Falsly predicted
            if predicted_bool[i]:
                #Network predicts signal but there is none in the data
                l[3] += 1
            else:
                #Network predicts no signal but there is one in the data
                l[2] += 1
        
        if abs(res[1][i][0] - 0.5) < 0.1:
            l[4] += 1
    
    plot_path = os.path.join(saves_path, net_name + '_removed_negatives.png')
    
    #Do the necessary plots
    x_pt_1 = [pt[0] for i, pt in enumerate(te_l) if predicted_bool[i]]
    x_pt_2 = [pt for i, pt in enumerate(te_c) if predicted_bool[i]]
    y_pt = [pt[0] for i, pt in enumerate(res[0]) if predicted_bool[i]]
    
    _do_plot(net_name, np.array(x_pt_1), np.array(x_pt_2), np.array(y_pt), plot_path, show=show)
    
    
    return(l)
    
