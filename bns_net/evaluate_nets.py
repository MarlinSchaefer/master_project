import keras
import numpy as np
from aux_functions import filter_keys, date_to_file_string
import os
from make_snr_plot import plot_true_and_calc_from_file
from evaluate_dual_output import evaluate_dual_output_form
from store_test_results import store_test_results
from store_test_results import store_full_results
from metrics import plot_false_alarm, plot_sensitivity, plot_false_alarm_prob, plot_sensitivity_prob
import imp
from ini_handeling import evaluate_net_defaults
from loss_plot import make_loss_plot

testing_file_names = ['testing_set_injections_100000_1.hf5', 'testing_set_injections_100000_2.hf5', 'testing_set_injections_100000_3.hf5', 'testing_set_injections_100000_4.hf5', 'testing_set_injections_100000_5.hf5']

def evaluate_training(net_name, dobj, dir_path, t_start, batch_size=32, **kwargs):
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
    
    store_test_results(net_last, dobj, prediction_path_last, batch_size=batch_size)
    
    prediction_path_best = ''
    
    if not net_best == None:
        prediction_path_best = os.path.join(dir_path, net_name + '_predictions_best_epoch_' + t_string + '.hf5')
        
        store_test_results(net_best, dobj, prediction_path_best, batch_size=batch_size)
    
    #Create loss plot
    loss_plot_path = os.path.join(dir_path, net_name + '_loss_plot_last_epoch_' + t_string + '.png')
    make_loss_plot(os.path.join(dir_path, net_name + "_results.json"), loss_plot_path)
    
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

#def evaluate_training_on_testing(net_name, dobj, dir_path, t_start, batch_size=32, testing_files=None, **kwargs):
    #"""Creates multiple important plots for the last and the best epoch.
    
    #Arguments
    #---------
    #net_name : str
        #The name of the network. There needs to be a net_name.hf5 file and a
        #net_name.py file.
    #dobj : object
        #An instance of a DataSet object from the module data_object. It needs to contain
        #all the data needed. (the data also needs to be loaded)
    #dir_path : str
        #String giving the path were all the data is supposed to be stored.
    #t_start : object
        #A datetime.datetime object containing the training starting time.
    #batch_size : int
        #The batch_size that is used in the predict_generator function.
    
    #Returns
    #-------
    #loss_plot_path : str
        #The path were the loss plot is saved.
    #SNR_plot_path_last : str
        #The path were the SNR plot for the last epoch is saved.
    #false_alarm_plot_path_last : str
        
    #sensitivity_plot_path_last : str
        
    #SNR_plot_path_best : str
        
    #false_alarm_plot_path_best : str
        
    #sensitivity_plot_path_best : str
        
    #wiki_data : dic
        #A dictironary containing all options to this function. (e.g. show_SNR_plot)
    #"""
    #opt_arg, kwargs = filter_keys(evaluate_net_defaults(), kwargs)
    
    #wiki_data = {}
    #for k, v in opt_arg.items():
        #wiki_data[k] = str(v)
    
    #t_string = date_to_file_string(t_start)
        
    #net_last = keras.models.load_model(os.path.join(dir_path, net_name + '.hf5'))
    
    ##Load networks
    #if not opt_arg['best_epoch'] == 0:
        #net_best = keras.models.load_model(os.path.join(dir_path, net_name + '_epoch_' + str(opt_arg['best_epoch']) + '.hf5'))
    #else:
        #net_best = None
    
    ##Run predict generator on the test data for each net.
    #prediction_path_last = os.path.join(dir_path, net_name + '_predictions_last_epoch_' + t_string + '.hf5')
    
    #store_test_results(net_last, dobj, prediction_path_last, batch_size=batch_size)
    
    #prediction_path_best = ''
    
    #if not net_best == None:
        #prediction_path_best = os.path.join(dir_path, net_name + '_predictions_best_epoch_' + t_string + '.hf5')
        
        #store_test_results(net_best, dobj, prediction_path_best, batch_size=batch_size)
    
    ##Create loss plot
    #loss_plot_path = os.path.join(dir_path, net_name + '_loss_plot_last_epoch_' + t_string + '.png')
    #make_loss_plot(os.path.join(dir_path, net_name + "_results.json"), loss_plot_path)
    
    ##Make SNR plots
    #SNR_plot_path_last = os.path.join(dir_path, net_name + '_snr_plot_last_epoch_' + t_string + '.png')
    
    #plot_true_and_calc_from_file(prediction_path_last, dobj, SNR_plot_path_last, show=opt_arg['show_snr_plot'], net_name=net_name + ' last epoch')
    
    #SNR_plot_path_best = ''
    
    #if not net_best == None:
        #SNR_plot_path_best = os.path.join(dir_path, net_name + '_snr_plot_best_epoch_' + t_string + '.png')
        
        #plot_true_and_calc_from_file(prediction_path_best, dobj, SNR_plot_path_best, show=opt_arg['show_snr_plot'], net_name=net_name + ' best epoch')
    
    ##Make false alarm plots
    #false_alarm_plot_path_last = os.path.join(dir_path, net_name + '_false_alarm_plot_last_epoch_' + t_string + '.png')
    
    #tmp_false_alarm_path_last = plot_false_alarm(dobj, prediction_path_last, false_alarm_plot_path_last, show=opt_arg['show_false_alarm'])
    
    #false_alarm_plot_prob_path_last = os.path.join(dir_path, net_name + '_false_alarm_plot_prob_last_epoch_' + t_string + '.png')
    
    #tmp_false_alarm_prob_path_last = plot_false_alarm_prob(dobj, prediction_path_last, false_alarm_plot_prob_path_last, show=opt_arg['show_false_alarm'])
    
    #false_alarm_plot_path_best = ''
    
    #false_alarm_plot_prob_path_best = ''
    
    #tmp_false_alarm_path_best = ''
    
    #tmp_false_alarm_prob_path_best = ''
    
    #if not net_best == None:
        #false_alarm_plot_path_best = os.path.join(dir_path, net_name + '_false_alarm_plot_best_epoch_' + t_string + '.png')
        
        #false_alarm_plot_prob_path_best = os.path.join(dir_path, net_name + '_false_alarm_plot_prob_best_epoch_' + t_string + '.png')
        
        #tmp_false_alarm_path_best = plot_false_alarm(dobj, prediction_path_best, false_alarm_plot_path_best, show=opt_arg['show_false_alarm'])
        
        #tmp_false_alarm_prob_path_best = plot_false_alarm_prob(dobj, prediction_path_best, false_alarm_plot_prob_path_best, show=opt_arg['show_false_alarm'])
    
    ##Make sensitivity plots
    #snr_range = dobj.get_file_properties()['snr']
    
    #sensitivity_plot_path_last = os.path.join(dir_path, net_name + '_sensitivity_plot_last_epoch_' + t_string + '.png')
    
    #sensitivity_plot_prob_path_last = os.path.join(dir_path, net_name + '_sensitivity_plot_prob_last_epoch_' + t_string + '.png')
    
    #plot_sensitivity(dobj, prediction_path_last, tmp_false_alarm_path_last, sensitivity_plot_path_last, bins=(snr_range[0], snr_range[1], 1), show=opt_arg['show_sensitivity_plot'])
    
    #plot_sensitivity_prob(dobj, prediction_path_last, tmp_false_alarm_prob_path_last, sensitivity_plot_prob_path_last, show=opt_arg['show_sensitivity_plot'])
    
    #sensitivity_plot_path_best = ''
    
    #sensitivity_plot_prob_path_best = ''
    
    #if not net_best == None:
        #sensitivity_plot_path_best = os.path.join(dir_path, net_name + '_sensitivity_plot_best_epoch_' + t_string + '.png')
        
        #sensitivity_plot_prob_path_best = os.path.join(dir_path, net_name + '_sensitivity_plot_prob_best_epoch_' + t_string + '.png')
        
        #plot_sensitivity(dobj, prediction_path_best, tmp_false_alarm_path_best, sensitivity_plot_path_best, bins=(snr_range[0], snr_range[1], 1), show=opt_arg['show_sensitivity_plot'])
        
        #plot_sensitivity_prob(dobj, prediction_path_best, tmp_false_alarm_prob_path_best, sensitivity_plot_prob_path_best, show=opt_arg['show_sensitivity_plot'])
    
    #if testing_files == None:
        #global testing_file_names
        #testing_files = testing_file_names
    
    #tmp_files = []
    
    #for f in testing_files:
        #if os.path.isfile(os.path.join(dir_path, f)):
            #tmp_files.append(f)
    
    #testing_files = tmp_files
    
    
    
    #if not testing_files == []:
        
    
    #return((loss_plot_path, SNR_plot_path_last, false_alarm_plot_path_last, false_alarm_plot_prob_path_last, sensitivity_plot_path_last, sensitivity_plot_prob_path_last, SNR_plot_path_best, false_alarm_plot_path_best, false_alarm_plot_prob_path_best, sensitivity_plot_path_best, sensitivity_plot_prob_path_best, wiki_data))

##def main():
    ##file_path = get_store_path()
    ##data_name = 'mult_output_data_medium_small.hf5'
    ###file_names = [str('collect_inception_net_6_rev_2_epoch_84.hf5'), str('collect_inception_net_6_rev_3_epoch_72.hf5'), str('collect_inception_net_6_rev_4_epoch_138.hf5')]
    
    ###for i, file_name in enumerate(file_names):
        ###print(i)
        ###net = keras.models.load_model(os.path.join(file_path, file_name))
        ###plot_name = file_name[:-4] + '_snr_plot.png'
        ###print('Plotted as {}'.format(plot_name))
        ###plot_true_and_calc_partial(net, os.path.join(file_path, data_name), os.path.join(file_path, plot_name), os.path.join(file_path, file_name[:29] + '.py'), net_name=file_name[:20])
    
    ###return
    ###net_name = 'collect_inception_net_6_rev_5_epoch_87'
    ###file_name = 'mult_output_data_medium_small'
    ###org_name = 'collect_inception_net_6_rev_5'
    ###print(evaluate_dual_output_form(net_name, file_name, org_name, screen_name='rev_5'))
    ##file_name = 'collect_inception_net_6_rev_5_epoch_87.hf5'
    ##net = keras.models.load_model(os.path.join(file_path, file_name))
    ##plot_name = file_name[:-4] + '_snr_plot.png'
    ##print('Plotted as {}'.format(plot_name))
    ##plot_true_and_calc_partial(net, os.path.join(file_path, data_name), os.path.join(file_path, plot_name), os.path.join(file_path, file_name[:29] + '.py'), net_name=file_name[:20])

##main()
