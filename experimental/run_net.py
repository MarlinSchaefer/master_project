import os
import keras
import importlib
from load_data import load_data
import numpy as np

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

def get_templates_path():
    return(get_store_path())

def get_net_path():
    return(get_store_path())

def net_exists_q(name, path=get_net_path()):
    return(os.path.isfile(os.path.join(path, name, ".hf5")))

def template_exists_q(name, path=get_templates_path()):
    return(os.path.isfile(os.path.join(path, name, ".hf5")))

def input_to_bool(string):
    true = ['y', 'Y', 'Yes', 'yes', 'Ja', 'ja', 'J', 'j', 'true', 'True', 'T', 't']
    return(string in true)

def run_net(net_name, temp_name, **kwargs):
    ignored_error = False
    
    opt_arg = {}
    
    #Properties for this function
    opt_arg['net_path'] = get_net_path()
    opt_arg['temp_path'] = get_templates_path()
    opt_arg['ignore_fixable_errors'] = False
    opt_arg['loss'] = 'mean_squared_error'
    opt_arg['optimizer'] = 'adam'
    opt_arg['metrics'] = ['mape']
    opt_arg['epochs'] = 10
    
    #Properties for the waveform itself
    opt_arg['approximant'] = "SEOBNRv4_opt"
    opt_arg['mass1'] = 30.0
    opt_arg['mass2'] = 30.0
    opt_arg['delta_t'] = 1.0 / 4096
    opt_arg['f_lower'] = 20.0
    
    #Properties for the generating program
    opt_arg['gw_prob'] = 1.0
    opt_arg['snr_low'] = 1.0
    opt_arg['snr_upper'] = 12.0
    opt_arg['resample_delta_t'] = 1.0 / 1024
    opt_arg['t_len'] = 64.0
    opt_arg['resample_t_len'] = 4.0
    opt_arg['num_of_templates'] = 20000
    opt_arg['random_starting_time'] = True
    opt_arg['random_phase'] = True
    
    for key in opt_arg.keys():
        if key is in kwargs:
            opt_arg[key] = kwargs.get(key)
            del kwargs[key]
    
    net_path = opt_arg['net_path']
    temp_path = opt_arg['temp_path']
    
    if not net_exists_q(net_name, path=net_path):
        try:
            #NOTE: The module needs to have a method 'get_model'
            net_mod = importlib.import_module(str(net_name))
            net = net_mod.get_model()
            
        except ImportError:
            raise NameError('There is no net named %s in %s.' % (net_name, net_path))
            return()
    else:
        #NOTE: The Net has to be stored with the ending 'hf5'
        net = keras.models.load_model(os.path.join(net_path, net_name, '.hf5'))
    
    input_layer_shape = (net.layers)[0].get_input_at(0).get_shape().as_list()
    output_layer_shape = (net.layers)[-1].get_output_at(0).get_shape().as_list()
    
    #Handle not existing template file
    #Either create or quit
    if not template_exists_q(temp_name, path=temp_path):
        if not opt_arg['ignore_fixable_errors']:
            inp = raw_input('No template file named %s found at %s.\nDo you want to generate it?\n' % (temp_name, temp_path))
        else:
            inp = 'y'
            ignored_error = True
        
        if input_to_bool(inp):
            if 'temp_creation_script' is in kwargs:
                temp_creation_script = kwargs.get('temp_creation_script')
                del kwargs['temp_creation_script']
                try:
                    #The custom module needs to have a 'create_file' method
                    custom_temp_script = importlib.import_module(str(temp_creation_script))
                    
                    custom_temp_script.create_file(name=temp_name, path=temp_path, approximant=opt_arg['approximant'], mass1=opt_arg['mass1'], mass2=opt_arg['mass2'], delta_t=opt_arg['delta_t'], f_lower=opt_arg['f_lower'], gw_prob=opt_arg['gw_prob'], snr_low=opt_arg['snr_low'], snr_upper=opt_arg['snr_upper'], resample_delta_t=opt_arg['resample_delta_t'], t_len=opt_arg['t_len'], resample_t_len=opt_arg['resample_t_len'], num_of_templates=opt_arg['num_of_templates'], random_starting_time=opt_arg['random_starting_time'], random_phase=opt_arg['random_phase'],**kwargs)#TODO: Input all optional arguments from above in here and the kwargs. Also the final file name
                except ImportError:
                    print("Could not import the creation file.")
                    return()
            else:
                try:
                    from make_template_bank import create_file
                    
                    create_file(name=temp_name, path=temp_path, approximant=opt_arg['approximant'], mass1=opt_arg['mass1'], mass2=opt_arg['mass2'], delta_t=opt_arg['delta_t'], f_lower=opt_arg['f_lower'], gw_prob=opt_arg['gw_prob'], snr_low=opt_arg['snr_low'], snr_upper=opt_arg['snr_upper'], resample_delta_t=opt_arg['resample_delta_t'], t_len=opt_arg['t_len'], resample_t_len=opt_arg['resample_t_len'], num_of_templates=opt_arg['num_of_templates'], random_starting_time=opt_arg['random_starting_time'], random_phase=opt_arg['random_phase'],**kwargs)#TODO: Input all optional arguments from above in here and the kwargs. Also the final file name
                except ImportError:
                    print("Could not import module 'make_template_file'")
                    return()
        else:
            return()
    
    #Load templates
    (train_data, train_labels), (test_data, test_labels) = load_data(os.path.join(temp_path, temp_name, ".hf5"))
    
    #Check sizes of loaded data, if data was not created by this function
    do_reshape = False
    if not train_data[0].shape == input_layer_shape:
        if not opt_arg['ignore_fixable_errors']:
            inp = 'False'
            inp = raw_input("The provided data does not fit the provided net.\nDo you want to try and reshape the data?\n")
            if input_to_bool(inp):
                do_reshape = True
        else:
            do_reshape = True
            ignored_error = True
    
    if do_reshape:
        cache = list(input_layer_shape)
        cache.prepend(len(train_data))
        np.reshape(train_data, tuple(cache))
        
        cache[0] = len(test_data)
        np.reshape(test_data, tuple(cache))
    
    do_reshape = False
    if not train_labels[0].shape == output_layer_shape:
        if not opt_arg['ignore_fixable_errors']:
            inp = 'False'
            inp = raw_input("The provided labels do not fit the provided net.\nDo you want to try and reshape the labels?\n")
            if input_to_bool(inp):
                do_reshape = True
        else:
            do_reshape = True
            ignored_error = True
    
    if do_reshape:
        cache = list(output_layer_shape)
        cache.prepend(len(train_labels))
        np.reshape(train_labels, tuple(cache))
        
        cache[0] = len(test_labels)
        np.reshape(test_labels, tuple(cache))
    
    #If everything is fine, train and evaluate the net
    net.compile(loss=opt_arg['loss'], optimizer=opt_arg['optimizer'], metrics=opt_arg['metrics'])
    
    net.fit(train_data, train_labels, epochs=opt_arg['epochs'])
    
    net.save(os.path.join(net_path, net_name + '_run_net_new.hf5'))
    
    print(net.evaluate(test_data, test_labels))
    
    if ignored_error:
        print(bcolors.WARNING + "This run ignored errors along the way!" + bcolors.ENDC)
