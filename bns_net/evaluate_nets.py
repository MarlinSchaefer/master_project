import keras
import numpy as np
from aux_functions import filter_keys, date_to_file_string
import os
from make_snr_plot import plot_true_and_calc_from_file
from evaluate_dual_output import evaluate_dual_output_form
from store_test_results import store_test_results, join_test_results
from metrics import plot_false_alarm, plot_sensitivity, plot_false_alarm_prob, plot_sensitivity_prob
import imp
from ini_handeling import evaluate_net_defaults
from loss_plot import make_loss_plot
import generator as g

testing_file_names = ['testing_set_injections_100000_1.hf5', 'testing_set_injections_100000_2.hf5', 'testing_set_injections_100000_3.hf5', 'testing_set_injections_100000_4.hf5', 'testing_set_injections_100000_5.hf5']

def evaluate_training(net_name, dobj, dir_path, t_start, batch_size=32, generator=g.DataGeneratorMultInput, **kwargs):
    """Creates multiple important plots for the last and the best epoch.
    
    Arguments
    ---------
    net_name : str
        The name of the network. There needs to be a net_name.hf5 file and a
        net_name.py file.
    dobj : object
        An instance of a DataSet object from the module data_object. It needs to contain
        all the data needed. (the data also needs to be loaded)
    dir_path : str
        String giving the path were all the data is supposed to be stored.
    t_start : object
        A datetime.datetime object containing the training starting time.
    batch_size : int
        The batch_size that is used in the predict_generator function.
    
    Returns
    -------
    loss_plot_path : str
        The path were the loss plot is saved.
    SNR_plot_path_last : str
        The path were the SNR plot for the last epoch is saved.
    false_alarm_plot_path_last : str
        
    sensitivity_plot_path_last : str
        
    SNR_plot_path_best : str
        
    false_alarm_plot_path_best : str
        
    sensitivity_plot_path_best : str
        
    wiki_data : dic
        A dictironary containing all options to this function. (e.g. show_SNR_plot)
    """
    opt_arg, kwargs = filter_keys(evaluate_net_defaults(), kwargs)
    
    wiki_data = {}
    for k, v in opt_arg.items():
        wiki_data[k] = str(v)
    
    t_string = date_to_file_string(t_start)
        
    net_last = keras.models.load_model(os.path.join(dir_path, net_name + '.hf5'))
    
    #Load networks
    if not opt_arg['best_epoch'] == 0:
        net_best = keras.models.load_model(os.path.join(dir_path, net_name + '_epoch_' + str(opt_arg['best_epoch']) + '.hf5'))
    else:
        net_best = None
    
    #Run predict generator on the test data for each net.
    prediction_path_last = os.path.join(dir_path, net_name + '_predictions_last_epoch_' + t_string + '.hf5')
    
    store_test_results(net_last, dobj, prediction_path_last, batch_size=batch_size, generator=generator)
    
    prediction_path_best = ''
    
    if not net_best == None:
        prediction_path_best = os.path.join(dir_path, net_name + '_predictions_best_epoch_' + t_string + '.hf5')
        
        store_test_results(net_best, dobj, prediction_path_best, batch_size=batch_size, generator=generator)
    
    #Create loss plot
    if opt_arg['make_loss_plot']:
        loss_plot_path = os.path.join(dir_path, net_name + '_loss_plot_last_epoch_' + t_string + '.png')
        make_loss_plot(os.path.join(dir_path, net_name + "_results.json"), loss_plot_path)
    else:
        loss_plot_path = 'N/A'
    
    #Make SNR plots
    SNR_plot_path_last = os.path.join(dir_path, net_name + '_snr_plot_last_epoch_' + t_string + '.png')
    
    plot_true_and_calc_from_file(prediction_path_last, dobj, SNR_plot_path_last, show=opt_arg['show_snr_plot'], net_name=net_name + ' last epoch')
    
    SNR_plot_path_best = ''
    
    if not net_best == None:
        SNR_plot_path_best = os.path.join(dir_path, net_name + '_snr_plot_best_epoch_' + t_string + '.png')
        
        plot_true_and_calc_from_file(prediction_path_best, dobj, SNR_plot_path_best, show=opt_arg['show_snr_plot'], net_name=net_name + ' best epoch')
    
    #Make false alarm plots
    false_alarm_plot_path_last = os.path.join(dir_path, net_name + '_false_alarm_plot_last_epoch_' + t_string + '.png')
    
    tmp_false_alarm_path_last = plot_false_alarm(dobj, prediction_path_last, false_alarm_plot_path_last, show=opt_arg['show_false_alarm'])
    
    false_alarm_plot_prob_path_last = os.path.join(dir_path, net_name + '_false_alarm_plot_prob_last_epoch_' + t_string + '.png')
    
    tmp_false_alarm_prob_path_last = plot_false_alarm_prob(dobj, prediction_path_last, false_alarm_plot_prob_path_last, show=opt_arg['show_false_alarm'])
    
    false_alarm_plot_path_best = ''
    
    false_alarm_plot_prob_path_best = ''
    
    tmp_false_alarm_path_best = ''
    
    tmp_false_alarm_prob_path_best = ''
    
    if not net_best == None:
        false_alarm_plot_path_best = os.path.join(dir_path, net_name + '_false_alarm_plot_best_epoch_' + t_string + '.png')
        
        false_alarm_plot_prob_path_best = os.path.join(dir_path, net_name + '_false_alarm_plot_prob_best_epoch_' + t_string + '.png')
        
        tmp_false_alarm_path_best = plot_false_alarm(dobj, prediction_path_best, false_alarm_plot_path_best, show=opt_arg['show_false_alarm'])
        
        tmp_false_alarm_prob_path_best = plot_false_alarm_prob(dobj, prediction_path_best, false_alarm_plot_prob_path_best, show=opt_arg['show_false_alarm'])
    
    #Make sensitivity plots
    snr_range = dobj.get_file_properties()['snr']
    
    sensitivity_plot_path_last = os.path.join(dir_path, net_name + '_sensitivity_plot_last_epoch_' + t_string + '.png')
    
    sensitivity_plot_prob_path_last = os.path.join(dir_path, net_name + '_sensitivity_plot_prob_last_epoch_' + t_string + '.png')
    
    plot_sensitivity(dobj, prediction_path_last, tmp_false_alarm_path_last, sensitivity_plot_path_last, bins=(snr_range[0], snr_range[1], 1), show=opt_arg['show_sensitivity_plot'])
    
    plot_sensitivity_prob(dobj, prediction_path_last, tmp_false_alarm_prob_path_last, sensitivity_plot_prob_path_last, show=opt_arg['show_sensitivity_plot'])
    
    sensitivity_plot_path_best = ''
    
    sensitivity_plot_prob_path_best = ''
    
    if not net_best == None:
        sensitivity_plot_path_best = os.path.join(dir_path, net_name + '_sensitivity_plot_best_epoch_' + t_string + '.png')
        
        sensitivity_plot_prob_path_best = os.path.join(dir_path, net_name + '_sensitivity_plot_prob_best_epoch_' + t_string + '.png')
        
        plot_sensitivity(dobj, prediction_path_best, tmp_false_alarm_path_best, sensitivity_plot_path_best, bins=(snr_range[0], snr_range[1], 1), show=opt_arg['show_sensitivity_plot'])
        
        plot_sensitivity_prob(dobj, prediction_path_best, tmp_false_alarm_prob_path_best, sensitivity_plot_prob_path_best, show=opt_arg['show_sensitivity_plot'])
    
    return((loss_plot_path, SNR_plot_path_last, false_alarm_plot_path_last, false_alarm_plot_prob_path_last, sensitivity_plot_path_last, sensitivity_plot_prob_path_last, SNR_plot_path_best, false_alarm_plot_path_best, false_alarm_plot_prob_path_best, sensitivity_plot_path_best, sensitivity_plot_prob_path_best, wiki_data))

def evaluate_training_on_testing(net_name, dobj, dir_path, t_start, batch_size=32, generator=g.DataGeneratorMultInput ,testing_files=None, **kwargs):
    """Creates multiple important plots for the last and the best epoch.
    
    Arguments
    ---------
    net_name : str
        The name of the network. There needs to be a net_name.hf5 file and a
        net_name.py file.
    dobj : object
        An instance of a DataSet object from the module data_object. It needs to contain
        all the data needed. (the data also needs to be loaded)
    dir_path : str
        String giving the path were all the data is supposed to be stored.
    t_start : object
        A datetime.datetime object containing the training starting time.
    batch_size : int
        The batch_size that is used in the predict_generator function.
    
    Returns
    -------
    loss_plot_path : str
        The path were the loss plot is saved.
    SNR_plot_path_last : str
        The path were the SNR plot for the last epoch is saved.
    false_alarm_plot_path_last : str
        
    sensitivity_plot_path_last : str
        
    SNR_plot_path_best : str
        
    false_alarm_plot_path_best : str
        
    sensitivity_plot_path_best : str
        
    wiki_data : dic
        A dictironary containing all options to this function. (e.g. show_SNR_plot)
    """
    opt_arg, kwargs = filter_keys(evaluate_net_defaults(), kwargs)
    
    wiki_data = {}
    for k, v in opt_arg.items():
        wiki_data[k] = str(v)
    
    t_string = date_to_file_string(t_start)
    
    ###
    
    if testing_files == None:
        global testing_file_names
        testing_files = testing_file_names
    
    tmp_files = []
    
    for f in testing_files:
        if os.path.isfile(os.path.join(dir_path, f)):
            tmp_files.append(f)
    
    testing_files = tmp_files
    
    ###
    
    net_last = keras.models.load_model(os.path.join(dir_path, net_name + '.hf5'))
    
    #Load networks
    if not opt_arg['best_epoch'] == 0:
        net_best = keras.models.load_model(os.path.join(dir_path, net_name + '_epoch_' + str(opt_arg['best_epoch']) + '.hf5'))
    else:
        net_best = None
    
    #Run predict generator on the test data for each net.
    tmp_prediction_paths_last = []
    tmp_prediction_paths_best = []
    for f in testing_files:
        tmp_prediction_paths_last.append(os.path.join(dir_path, os.splittext(f)[0] + '_predictions_last.hf5'))
        if not net_best == None:
            tmp_prediction_paths_best.append(os.path.join(dir_path, os.splittext(f)[0] + '_predictions_best.hf5'))
        
        dobj.set_file_name(f)
        dobj.unload_all()
        dobj.get_set()
        
        store_test_results(net_last, dobj, tmp_prediction_paths_last[-1], batch_size=batch_size, generator=generator)
        if not net_best == None:
            store_test_results(net_best, dobj, tmp_prediction_paths_best, batch_size=batch_size, generator=generator)
    
    prediction_path_last = os.path.join(dir_path, net_name + '_predictions_last_epoch_full_testing_' + t_string + '.hf5')
    join_test_results(tmp_prediction_paths_last, prediction_path_last, delete_copied_files=True)
    prediction_path_best = ''
    if not net_best == None:
        prediction_path_best = os.path.join(dir_path, net_name + '_predictions_best_epoch_full_testing_' + t_string + '.hf5')
        join_test_results(tmp_prediction_paths_best, prediction_path_best, delete_copied_files=True)
    
    #Make SNR plots
    SNR_plot_path_last = os.path.join(dir_path, net_name + '_snr_plot_last_epoch_full_testing_' + t_string + '.png')
    
    plot_true_and_calc_from_file(prediction_path_last, dobj, SNR_plot_path_last, show=opt_arg['show_snr_plot'], net_name=net_name + ' last epoch')
    
    SNR_plot_path_best = ''
    
    if not net_best == None:
        SNR_plot_path_best = os.path.join(dir_path, net_name + '_snr_plot_best_epoch_full_testing_' + t_string + '.png')
        
        plot_true_and_calc_from_file(prediction_path_best, dobj, SNR_plot_path_best, show=opt_arg['show_snr_plot'], net_name=net_name + ' best epoch')
    
    #Make false alarm plots
    false_alarm_plot_path_last = os.path.join(dir_path, net_name + '_false_alarm_plot_last_epoch_full_testing_' + t_string + '.png')
    
    tmp_false_alarm_path_last = plot_false_alarm(dobj, prediction_path_last, false_alarm_plot_path_last, show=opt_arg['show_false_alarm'])
    
    false_alarm_plot_prob_path_last = os.path.join(dir_path, net_name + '_false_alarm_plot_prob_last_epoch_full_testing_' + t_string + '.png')
    
    tmp_false_alarm_prob_path_last = plot_false_alarm_prob(dobj, prediction_path_last, false_alarm_plot_prob_path_last, show=opt_arg['show_false_alarm'])
    
    false_alarm_plot_path_best = ''
    
    false_alarm_plot_prob_path_best = ''
    
    tmp_false_alarm_path_best = ''
    
    tmp_false_alarm_prob_path_best = ''
    
    if not net_best == None:
        false_alarm_plot_path_best = os.path.join(dir_path, net_name + '_false_alarm_plot_best_epoch_full_testing_' + t_string + '.png')
        
        false_alarm_plot_prob_path_best = os.path.join(dir_path, net_name + '_false_alarm_plot_prob_best_epoch_full_testing_' + t_string + '.png')
        
        tmp_false_alarm_path_best = plot_false_alarm(dobj, prediction_path_best, false_alarm_plot_path_best, show=opt_arg['show_false_alarm'])
        
        tmp_false_alarm_prob_path_best = plot_false_alarm_prob(dobj, prediction_path_best, false_alarm_plot_prob_path_best, show=opt_arg['show_false_alarm'])
    
    #Make sensitivity plots
    snr_range = dobj.get_file_properties()['snr']
    
    sensitivity_plot_path_last = os.path.join(dir_path, net_name + '_sensitivity_plot_last_epoch_full_testing_' + t_string + '.png')
    
    sensitivity_plot_prob_path_last = os.path.join(dir_path, net_name + '_sensitivity_plot_prob_last_epoch_full_testing_' + t_string + '.png')
    
    plot_sensitivity(dobj, prediction_path_last, tmp_false_alarm_path_last, sensitivity_plot_path_last, bins=(snr_range[0], snr_range[1], 1), show=opt_arg['show_sensitivity_plot'])
    
    plot_sensitivity_prob(dobj, prediction_path_last, tmp_false_alarm_prob_path_last, sensitivity_plot_prob_path_last, show=opt_arg['show_sensitivity_plot'])
    
    sensitivity_plot_path_best = ''
    
    sensitivity_plot_prob_path_best = ''
    
    if not net_best == None:
        sensitivity_plot_path_best = os.path.join(dir_path, net_name + '_sensitivity_plot_best_epoch_full_testing_' + t_string + '.png')
        
        sensitivity_plot_prob_path_best = os.path.join(dir_path, net_name + '_sensitivity_plot_prob_best_epoch_full_testing_' + t_string + '.png')
        
        plot_sensitivity(dobj, prediction_path_best, tmp_false_alarm_path_best, sensitivity_plot_path_best, bins=(snr_range[0], snr_range[1], 1), show=opt_arg['show_sensitivity_plot'])
        
        plot_sensitivity_prob(dobj, prediction_path_best, tmp_false_alarm_prob_path_best, sensitivity_plot_prob_path_best, show=opt_arg['show_sensitivity_plot'])
    
    return((SNR_plot_path_last, false_alarm_plot_path_last, false_alarm_plot_prob_path_last, sensitivity_plot_path_last, sensitivity_plot_prob_path_last, SNR_plot_path_best, false_alarm_plot_path_best, false_alarm_plot_prob_path_best, sensitivity_plot_path_best, sensitivity_plot_prob_path_best))
