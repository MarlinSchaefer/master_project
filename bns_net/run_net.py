import os
import keras
from load_data import load_data, load_parameter_space, load_calculated_snr, load_testing_labels
import numpy as np
import imp
from loss_plot import make_loss_plot
import time
from wiki import make_wiki_entry, read_json, model_to_string
from ini_handeling import run_net_defaults, load_options
from store_test_results import store_test_results
from metrics import plot_false_alarm, plot_sensitivity, joint_snr_bar_plot, joint_snr_false_alarm_plot, joint_prob_false_alarm_plot, joint_prob_bar_plot
import traceback

"""
TODO:
-Add support for .ini files
"""

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_store_path():
    return(os.path.join(os.path.dirname(os.path.abspath(__file__)), "saves"))

from make_snr_plot import plot_true_and_calc_partial, plot_true_and_calc_from_file

def get_templates_path():
    return(get_store_path())

def get_net_path():
    return(get_store_path())

def net_exists_q(name, path=get_net_path()):
    return(os.path.isfile(os.path.join(path, name + ".hf5")))

def template_exists_q(name, path=get_templates_path()):
    return(os.path.isfile(os.path.join(path, name + ".hf5")))

def input_to_bool(string):
    true = ['y', 'Y', 'Yes', 'yes', 'Ja', 'ja', 'J', 'j', 'true', 'True', 'T', 't', '1']
    return(string in true)

def filter_keys(opt_arg, kwargs):
    for key in opt_arg.keys():
        if key in kwargs:
            opt_arg[key] = kwargs.get(key)
            del kwargs[key]
    
    return(opt_arg, kwargs)

def set_template_file(temp_name, temp_path, args):
    opt_arg = args[0]
    kwargs = args[1]
    ignored_error = False
    
    if not template_exists_q(temp_name, path=temp_path) or opt_arg['overwrite_template_file']:
        if not opt_arg['ignore_fixable_errors'] and not opt_arg['overwrite_template_file']:
            inp = raw_input('No template file named %s found at %s.\nDo you want to generate it?\n' % (temp_name, temp_path))
        else:
            inp = 'y'
            if not opt_arg['overwrite_template_file']:
                ignored_error = True
        
        if input_to_bool(inp):
            if 'temp_creation_script' in kwargs:
                temp_creation_script = kwargs.get('temp_creation_script')
                del kwargs['temp_creation_script']
                try:
                    #The custom module needs to have a 'create_file' method
                    custom_temp_script = importlib.import_module(str(temp_creation_script))
                    
                    custom_temp_script.create_file(name=temp_name, path=temp_path, **kwargs)
                except ImportError:
                    print("Could not import the creation file.")
                    exit()
            else:
                try:
                    #from make_template_bank_v2 import create_file
                    from make_template_bank_bns import create_file
                    
                    create_file(name=temp_name, path=temp_path, **kwargs)
                except ImportError:
                    print("Could not import module 'make_template_bank_new'")
                    exit()
        else:
            exit()
    
    return(ignored_error)

def _train_net(net, data_path, **opt_arg):
    """Function to handle different ways of training the network.
    
    
    
    """
    net_path = opt_arg['net_path']
    net_name = opt_arg['net_name']
    store_results_path = opt_arg['store_results_path']
    hist = None
    
    #If everything is fine, train and evaluate the net
    if opt_arg['use_custom_compilation']:
        try:
            #NOTE: The module needs to have a method 'train_model', which returns the trained model.
            print("Using custom compilation function")
            net_mod = imp.load_source("net_mod", str(os.path.join(net_path, net_name + '.py')))
            net_mod.compile_model(net)
        except IOError:
            raise NameError('There is no net named %s in %s.' % (net_name, net_path))
            return()
    else:
        net.compile(loss=opt_arg['loss'], optimizer=opt_arg['optimizer'], metrics=opt_arg['metrics'])
        
    print(net.summary())
    if opt_arg['use_custom_train_function']:
        try:
            if not opt_arg['use_data_object']:
                #NOTE: The module needs to have a method 'train_model', which returns the trained model.
                net_mod = imp.load_source("net_mod", str(os.path.join(net_path, net_name + '.py')))
                net = net_mod.train_model(net, data_path, store_results_path, epochs=opt_arg['epochs'], epoch_break=opt_arg['epoch_break'], batch_size=opt_arg['batch_size'])
            else:
                net_mod = imp.load_source("net_mod", str(os.path.join(net_path, net_name + '.py')))
                print("opt_arg['store_results_path']: {}".format(opt_arg['store_results_path']))
                print("Store results path: {}".format(store_results_path))
                net = net_mod.train_model(net, opt_arg['dobj'], store_results_path, epochs=opt_arg['epochs'], epoch_break=opt_arg['epoch_break'], batch_size=opt_arg['batch_size'])
        except IOError:
            raise NameError('There is no net named %s in %s.' % (net_name, net_path))
            return()
    else:
        if not opt_arg['use_data_object']:
            (train_data, train_labels), (test_data, test_labels) = get_data(data_path, **opt_arg)
            hist = net.fit(train_data, train_labels, epochs=opt_arg['epochs'])
        else:
            train_data = opt_arg['dobj'].loaded_train_data
            train_labels = opt_arg['dobj'].loaded_train_labels
            hist = net.fit(train_data, train_labels, epochs=opt_arg['epochs'])
        
    net.save(os.path.join(store_results_path, net_name + '.hf5'))
        
    #print(net.evaluate(test_data, test_labels))
    
    return(hist)

def date_to_file_string(t):
    return("{}{}{}{}{}{}".format(t.tm_mday, t.tm_mon, t.tm_year, t.tm_hour, t.tm_min, t.tm_sec))

def get_data(data_path, **opt_arg):
    if opt_arg['format_data']:
        try:
            #NOTE: The module needs to have a method 'train_model', which returns the trained model.
            net_mod = imp.load_source("net_mod", str(os.path.join(opt_arg['net_path'], opt_arg['net_name'] + '.py')))
            return(net_mod.get_formatted_data(data_path))
        except IOError:
            raise NameError('There is no function named "get_formatted_data" in %s.py.' % (net_name))
            return()
    else:
        return(load_data(data_path))

from evaluate_nets import evaluate_training, evaluate_training_on_testing

"""
Function to handle running a neural net.

A neural net with the name specified
in 'net_name' has to exist in the folder given in 'net_path'. If not specified
otherwise, this path will default to the subdirectory 'saves'.
If no template_file with the name as specified in 'temp_name' exists, the user
is prompted to choose wether or not he or she wants to generate the
template-file before training the net. If not, the program will quit.
This function takes optional keyword-arguments. All keyword-arguments used by
this function explicitly (i.e. attributes, which are not being passed to other
functions) are listed below.
Alongside the arguments this function uses, it can take all arguments the
function 'create_file' from the module 'make_template_bank' takes. This
therefore includes all arguments which 'pycbc.waveform.get_td_waveform' can
take.

Args:
    -(str)net_name: Name of the neural net to use (without file-extension).
    -(str)temp_name: Name of the template file (without file-extension).
    -(op,bool)ignore_fixable_errors: Wether or not to prompt the user with
                                     choices about wether to fix an error or
                                     not (*). Default: False
    -(op,bool)overwrite_template_file: Wether to overwrite or use existing
                                       template-files. Default: False
    -(op,bool)overwrite_net_file: Wether to load or overwrite an existing
                                  network-model. Default: True
    -(op,str/func)loss: The loss-option from keras.models.fit().
                        Default: 'mean_squared_error'
    -(op,str/func)optimizer: The optimizer option from keras.models.fit().
                             Default: 'adam'
    -(op,str/func/list)metrics: The metrics option  from keras.models.fit().
                                Default: ['mape']
    -(op,int)epochs: The epochs option from keras.models.fit().
                     Default: 10
    -(op,bool)show_snr_plot: Wether or not to show a plot to visualize the
                             resulting net. An image is stored even when this
                             option is set to false. (**) Default: True
    -(op,bool)only_print_image: Wether or not to train the net or just print
                                the final evaluation image. Default: False
    -(op,bool)use_custom_train_function: Wether to use keras.model.fit or a
                                         custom training function. (3*)
                                         Default: False

Ret:
    -(void)

Notes:
    -(*): Fixable errors are a non-existing template file, as this can be
           generated prior to training the net, and wrong data-shapes.
    -(**): The location of the image will be the same as the neural_net
           location. It will be a .png file.
    -If this function is called and should create a new template file, it MUST
     be called like:
     
        if __name__ == "__main__":
            run_net(net_name, temp_name, **kwargs)
     
     as it utilizes the mulitprocessing.Pool module.
    -(3*): The custom training function needs to be provided in the network.py
           file, which would be used to load the network if no .hf5 stored
           version existis. This function needs to have the network as a
           parameter and needs to return the trained model.
"""
def run_net(net_name, temp_name, **kwargs):
    try:
        wiki_data = {}
        wiki_data['time_init'] = time.gmtime(time.time())
        t_string = date_to_file_string(wiki_data['time_init'])
        
        if 'ini_file' in kwargs and os.path.isfile(kwargs['ini_file']):
            kwargs, temp = filter_keys(load_options(kwargs['ini_file']), kwargs)
            kwargs.update(temp)
            del temp
        ignored_error = False
        
        opt_arg = {}
        
        #Properties for this function
        opt_arg['net_path'] = get_net_path()
        opt_arg['temp_path'] = get_templates_path()
        opt_arg['net_name'] = net_name
        opt_arg['temp_name'] = temp_name
        
        i = 0
        created_dir = False
        dir_path = ''
        while not created_dir:
            try:
                if i == 0:
                    dir_path = os.path.join(get_store_path(), net_name + '_' + t_string)
                else:
                    dir_path = os.path.join(get_store_path(), net_name + '_' + t_string + '(' + str(i) + ')')
                os.mkdir(dir_path)
                created_dir = True
            except OSError:
                i += 1
                pass
        
        opt_arg['store_results_path'] = dir_path
        
        opt_arg.update(run_net_defaults())
        
        #Replace some default values by None
        if opt_arg['dobj'] == False:
            opt_arg['dobj'] = None
        
        if opt_arg['slice_size'] == False:
            opt_arg['slice_size'] = None
        
        if opt_arg['data_slice'] == False:
            opt_arg['data_slice'] = None
        
        wiki_data['programm_internals'] = {}
        
        #Look for any key of the values above that was overwritten and delete it
        #from kwargs to pass kwargs on to other functions.
        opt_arg, kwargs = filter_keys(opt_arg, kwargs)
        
        #Store the optional arguments set in the wiki
        for key in opt_arg.keys():
            wiki_data['programm_internals'][key] = opt_arg[key]
        
        #Set shortcuts to the paths for easier reference
        net_path = opt_arg['net_path']
        temp_path = opt_arg['temp_path']
        
        #The standard training function 'fit' doesn't take None as an option for
        #the epochs. Hance check this here.
        if opt_arg['epochs'] == None and not opt_arg['use_custom_train_function']:
            raise ValueError('Cannot set "epochs" to "None" without using a custom training function which can handle this exception.')
            return
        
        #Load the network. If there is a .hf5 file named correctly and the option
        #'overwrite_net_file' is set to false load this one. Otherwise import the
        #model from a .py file.
        if not net_exists_q(net_name, path=net_path) or opt_arg['overwrite_net_file']:
            try:
                #NOTE: The module needs to have a method 'get_model'
                net_mod = imp.load_source("net_mod", str(os.path.join(net_path, net_name + '.py')))
                net = net_mod.get_model()
                
            except IOError:
                raise NameError('There is no net named %s in %s.' % (net_name, net_path))
                return()
        else:
            #NOTE: The Net has to be stored with the ending 'hf5'
            net = keras.models.load_model(os.path.join(net_path, net_name + '.hf5'))
        
        #Get the important shapes of the layers to possibly reshape the loaded
        #template data later on
        input_layer_shape = (net.layers)[0].get_input_at(0).get_shape().as_list()
        input_layer_shape = tuple(input_layer_shape[1:])
        output_layer_shape = (net.layers)[-1].get_output_at(0).get_shape().as_list()
        output_layer_shape = tuple(output_layer_shape[1:])
        
        kwargs['data_shape'] = input_layer_shape
        kwargs['label_shape'] = output_layer_shape
        
        #Handle not existing template file
        #Either create or quit
        wiki_data['template_generation'] = {}
        wiki_data['template_generation']['time_start'] = time.gmtime(time.time())
        
        if set_template_file(temp_name, temp_path, [opt_arg, kwargs]):
            ignored_error = True
        
        if opt_arg['generate_templates_only']:
            return
            
        wiki_data['template_generation']['time_end'] = time.gmtime(time.time())
        
        
        #Load templates
        full_template_path = os.path.join(temp_path, temp_name + ".hf5")
        print("Loading templates from: {}".format(full_template_path))
        if opt_arg['use_data_object']:
            net_mod = imp.load_source("net_mod", str(os.path.join(net_path, net_name + '.py')))
            opt_arg['dobj'] = net_mod.get_data_obj(full_template_path)
            dobj = opt_arg['dobj']
            if opt_arg['slice_size'] == None:
                if opt_arg['data_slice'] == None:
                    dobj.get_set()
                else:
                    dobj.get_set(slice=opt_arg['data_slice'])
            else:
                if not opt_arg['data_slice'] == None:
                    num_slices = int(np.ceil(float(opt_arg['data_slice']) / opt_arg['slice_size']))
                else:
                    num_slices= int(np.ceil(float(max(dobj.training_data_shape[0], dobj.testing_data_shape[0])) / opt_arg['slice_size']))
                
                for i in range(num_slices):
                    dobj.get_set(slice=(i*opt_arg['slice_size'], (i+1)*opt_arg['slice_size']))
            generator = net_mod.get_generator()
        
        #Training takes place here
        if not opt_arg['only_print_image']:
            wiki_data['training'] = {}
            wiki_data['training']['time_start'] = time.gmtime(time.time())
            
            hist = _train_net(net, full_template_path, **opt_arg)
            
            wiki_data['training']['time_end'] = time.gmtime(time.time())
            
        #Give a warning if some errors were ignored and not specifically said
        #(i.e. user input) to be automatically handeled
        if ignored_error:
            print(bcolors.WARNING + "This run ignored errors along the way!" + bcolors.ENDC)
        
        #Store wiki data about the loss
        try:
            wiki_data['loss'] = read_json(os.path.join(opt_arg['store_results_path'], net_name + "_results.json"))
            pass
        except (IOError, IndexError):
            if not hist == None:
                losses = hist.history['loss']
            else:
                losses = [None]
            
            wiki_data['loss'] = {}
            wiki_data['loss']['min_training'] = (0, np.inf)
            wiki_data['loss']['min_testing'] = (0, np.inf)
            wiki_data['loss']['last_training'] = (len(losses), losses[-1])
            wiki_data['loss']['last_testing'] = (0, np.inf)
            
            #Find minimum loss
            for idx, pt in enumerate(losses):
                if pt < wiki_data['loss']['min_training'][1]:
                    wiki_data['loss']['min_training'] = (idx, pt)
            
            print(bcolors.OKGREEN + 'Could not store loss history in the wiki, as the %s file could not be found.' % (net_name + '_results.json') + bcolors.ENDC)
        
        kwargs['best_epoch'] = wiki_data['loss']['min_testing'][0]
        
        #If something goes wrong during plotting, keep going
        try:
            #Create all plots
            wik = evaluate_training(net_name, dobj, opt_arg['store_results_path'], wiki_data['time_init'], batch_size=opt_arg['batch_size'], generator=generator, **kwargs)
            
            #Store data to wiki
            wiki_data['loss_plot_path'] = wik[0]
            wiki_data['SNR_plot_path_last_epoch'] = wik[1]
            wiki_data['false_alarm_plot_path_last_epoch'] = wik[2]
            wiki_data['false_alarm_plot_prob_path_last_epoch'] = wik[3]
            wiki_data['sensitivity_plot_path_last_epoch'] = wik[4]
            wiki_data['sensitivity_plot_prob_path_last_epoch'] = wik[5]
            wiki_data['SNR_plot_path_best_epoch'] = wik[6]
            wiki_data['false_alarm_plot_path_best_epoch'] = wik[7]
            wiki_data['false_alarm_plot_prob_path_best_epoch'] = wik[8]
            wiki_data['sensitivity_plot_path_best_epoch'] = wik[9]
            wiki_data['sensitivity_plot_prob_path_best_epoch'] = wik[10]
            wiki_data['p_value_distribution_plot_path_best_epoch'] = wik[11]
            wiki_data['p_value_distribution_plot_path_last_epoch'] = wik[12]
            wiki_data['plot_options'] = wik[13]
        except:
            traceback.print_exc()
            pass
        
        try:
            sens_last_path, ext = os.path.splitext(wiki_data['sensitivity_plot_path_last_epoch'])
            sens_best_path, ext = os.path.splitext(wiki_data['sensitivity_plot_path_best_epoch'])
            sens_prob_last_path = os.path.splitext(wiki_data['sensitivity_plot_prob_path_last_epoch'])[0]
            sens_prob_best_path = os.path.splitext(wiki_data['sensitivity_plot_prob_path_best_epoch'])[0]
            fa_snr_last_path, ext = os.path.splitext(wiki_data['false_alarm_plot_path_last_epoch'])
            fa_snr_best_path, ext = os.path.splitext(wiki_data['false_alarm_plot_path_best_epoch'])
            fa_prob_last_path, ext = os.path.splitext(wiki_data['false_alarm_plot_prob_path_last_epoch'])
            fa_prob_best_path, ext = os.path.splitext(wiki_data['false_alarm_plot_prob_path_best_epoch'])
            joint_sens_path = os.path.join(opt_arg['store_results_path'], 'joint_sensitivity_plot.png')
            joint_sens_prob_path = os.path.join(opt_arg['store_results_path'], 'joint_probabilitiy_sensitivity_plot.png')
            joint_false_alarm_plot_snr = os.path.join(opt_arg['store_results_path'], 'joint_false_alarm_plot_snr.png')
            joint_false_alarm_plot_prob = os.path.join(opt_arg['store_results_path'], 'joint_false_alarm_plot_prob.png')
            joint_snr_bar_plot(sens_last_path+'.hf5', sens_best_path+'.hf5', joint_sens_path)
            joint_prob_bar_plot(sens_prob_last_path+'.hf5', sens_prob_best_path+'.hf5', joint_sens_prob_path)
            joint_snr_false_alarm_plot(fa_snr_last_path+'.hf5', fa_snr_best_path+'.hf5', joint_false_alarm_plot_snr)
            joint_prob_false_alarm_plot(fa_prob_last_path+'.hf5', fa_prob_best_path+'.hf5', joint_false_alarm_plot_prob)
        except:
            traceback.print_exc()
            pass
        
        if opt_arg['evaluate_on_large_testing_set']:
            wik = evaluate_training_on_testing(net_name, dobj, opt_arg['store_results_path'], wiki_data['time_init'], batch_size=opt_arg['batch_size'], generator=generator, **kwargs)
        
            #Store data to wiki
            wiki_data['SNR_plot_path_last_epoch_full_testing'] = wik[0]
            wiki_data['false_alarm_plot_path_last_epoch_full_testing'] = wik[1]
            wiki_data['false_alarm_plot_prob_path_last_epoch_full_testing'] = wik[2]
            wiki_data['sensitivity_plot_path_last_epoch_full_testing'] = wik[3]
            wiki_data['sensitivity_plot_prob_path_last_epoch_full_testing'] = wik[4]
            wiki_data['SNR_plot_path_best_epoch_full_testing'] = wik[5]
            wiki_data['false_alarm_plot_path_best_epoch_full_testing'] = wik[6]
            wiki_data['false_alarm_plot_prob_path_best_epoch_full_testing'] = wik[7]
            wiki_data['sensitivity_plot_path_best_epoch_full_testing'] = wik[8]
            wiki_data['sensitivity_plot_prob_path_best_epoch_full_testing'] = wik[9]
        
        ##Plot the distribution of labels against predictions
        #if not opt_arg['use_data_object']:
            #train_calculated_snr, test_calculated_snr = load_calculated_snr(full_template_path)
            #unformatted_test_labels = load_testing_labels(full_template_path)
        #else:
            #train_calculated_snr = dobj.get_formatted_data('training', 'train_snr_calculated')
            #test_calculated_snr = dobj.get_formatted_data('testing', 'test_snr_calculated')
            #unformatted_test_labels = dobj.get_raw_data('testing', 'test_labels')
        
        #t_string = date_to_file_string(wiki_data['training']['time_start'])
        #wiki_data['SNR_plot_name'] = net_name + '_snr_' + t_string + '.png'
        #if opt_arg['use_data_object']:
            #result_file_path = os.path.join(get_store_path(), net_name + '_predictions_' + t_string + '.hf5')
            #store_test_results(net, dobj, result_file_path, batch_size=opt_arg['batch_size'])
            #plot_true_and_calc_from_file(result_file_path, dobj, os.path.join(net_path, wiki_data['SNR_plot_name']), show=opt_arg['show_snr_plot'], net_name=net_name)
            #wiki_data['false_alarm_plot_path'] = os.path.join(get_store_path(), net_name + '_false_alarm_plot_' + t_string + '.png')
            #false_alarm_path = plot_false_alarm(dobj, result_file_path, wiki_data['false_alarm_plot_path'], show=opt_arg['show_false_alarm'])
            #wiki_data['sensitivity_plot_path'] = os.path.join(get_store_path(), net_name + '_sensitivity_plot_' + t_string + '.png')
            #snr_range = dobj.get_file_properties()['snr']
            #plot_sensitivity(dobj, result_file_path, false_alarm_path, wiki_data['sensitivity_plot_path'], bins=(snr_range[0], snr_range[1], 1), show=opt_arg['show_sensitivity_plot'])
        #else:
            #plot_true_and_calc_partial(net, full_template_path, os.path.join(net_path, wiki_data['SNR_plot_name']), os.path.join(net_path, net_name + '.py'), batch_size=opt_arg['batch_size'], show=opt_arg['show_snr_plot'], net_name=net_name)
        
        ##Plot the loss over some recorded history
        #wiki_data['loss_plot_name'] = net_name + '_loss_plot_' + t_string + '.png'
        #make_loss_plot(os.path.join(get_store_path(), net_name + "_results.json"), os.path.join(get_store_path(), wiki_data['loss_plot_name']))
        ##try:
            ##wiki_data['loss_plot_name'] = net_name + '_loss_plot_' + t_string + '.png'
            ##make_loss_plot(os.path.join(get_store_path(), net_name + "_results.json"), os.path.join(get_store_path(), wiki_data['loss_plot_name']))
        ##except IOError:
            ##print(bcolors.OKGREEN + 'Could not create plot of the loss function, as the %s file could not be found.' % (net_name + '_results.json') + bcolors.ENDC)
        
        wiki_data['ignored_errors'] = ignored_error
        wiki_data['template_properties'] = load_parameter_space(full_template_path)
        wiki_data['network'] = model_to_string(net)
        wiki_data['custom_message'] = opt_arg['custom_message']
        
        #Create a wiki-entry
        #Call twice to store with the network and to append to the general file.
        if opt_arg['create_wiki_entry']:
            make_wiki_entry(wiki_data)
            make_wiki_entry(wiki_data, path=opt_arg['store_results_path'])
    except:
        if opt_arg['create_wiki_entry']:
            make_wiki_entry(wiki_data)
            make_wiki_entry(wiki_data, path=opt_arg['store_results_path'])
        traceback.print_exc()
