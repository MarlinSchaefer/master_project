import os
import keras
from load_data import load_data, load_parameter_space, load_calculated_snr
import numpy as np
import imp
from make_snr_plot import plot_true_and_calc
from loss_plot import make_loss_plot
import time
from wiki import make_wiki_entry, read_json, model_to_string
from ini_handeling import run_net_defaults, load_options

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

def reshape_data(train_data, test_data, final_shape, **opt_arg):
    #print('Train Data shape: {}'.format(train_data.shape))
    #print('Test Data shape: {}'.format(test_data.shape))
    #print('Final shape: {}'.format(final_shape))
    do_reshape = False
    ignored_error = False
    #print('Output shape: {}'.format(output_layer_shape))
    #print('data shape: {}'.format(train_labels.shape))
    if not train_data[0].shape == final_shape:
        if not opt_arg['ignore_fixable_errors']:
            inp = 'False'
            inp = raw_input("The provided data does not fit the provided net.\nDo you want to try and reshape the data?\n")
            if input_to_bool(inp):
                do_reshape = True
            else:
                exit()
        else:
            do_reshape = True
            ignored_error = True
    
    if do_reshape:
        cache = list(final_shape)
        cache.insert(0, len(train_data))
        np.reshape(train_data, tuple(cache))
        
        cache[0] = len(test_data)
        np.reshape(test_data, tuple(cache))
    
    return((train_data, test_data, ignored_error))

def _train_net(net, net_name, train_data, test_data, train_labels, test_labels, **opt_arg):
    """Function to handle different ways of training the network.
    
    
    
    """
    net_path = opt_arg['net_path']
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
            #NOTE: The module needs to have a method 'train_model', which returns the trained model.
            net_mod = imp.load_source("net_mod", str(os.path.join(net_path, net_name + '.py')))
            net = net_mod.train_model(net, train_data, train_labels, test_data, test_labels, net_path, epochs=opt_arg['epochs'], epoch_break=opt_arg['epoch_break'])
        except IOError:
            raise NameError('There is no net named %s in %s.' % (net_name, net_path))
            return()
    else:
        hist = net.fit(train_data, train_labels, epochs=opt_arg['epochs'])
        
    net.save(os.path.join(net_path, net_name + '.hf5'))
        
    print(net.evaluate(test_data, test_labels))
    
    return(hist)

def date_to_file_string(t):
    return("{}{}{}{}{}{}".format(t.tm_mday, t.tm_mon, t.tm_year, t.tm_hour, t.tm_min, t.tm_sec))

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
    if 'ini_file' in kwargs and os.path.isfile(kwargs['ini_file']):
        kwargs, temp = filter_keys(load_options(kwargs['ini_file']), kwargs)
        kwargs.update(temp)
        del temp
    ignored_error = False
    
    wiki_data = {}
    opt_arg = {}
    
    #Properties for this function
    opt_arg['net_path'] = get_net_path()
    opt_arg['temp_path'] = get_templates_path()
    opt_arg.update(run_net_defaults())
    #opt_arg['ignore_fixable_errors'] = False
    #opt_arg['loss'] = 'mean_squared_error'
    #opt_arg['optimizer'] = 'adam'
    #opt_arg['metrics'] = ['mape']
    #opt_arg['epochs'] = 10
    #opt_arg['overwrite_template_file'] = False
    #opt_arg['overwrite_net_file'] = True
    #opt_arg['show_snr_plot'] = True
    #opt_arg['only_print_image'] = False
    #opt_arg['use_custom_train_function'] = False
    #opt_arg['epoch_break'] = 10
    #opt_arg['create_wiki_entry'] = True
    
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
        
    wiki_data['template_generation']['time_end'] = time.gmtime(time.time())
    
    
    #Load templates
    print(os.path.join(temp_path, temp_name + ".hf5"))
    if opt_arg['format_data']:
        try:
            #NOTE: The module needs to have a method 'train_model', which returns the trained model.
            net_mod = imp.load_source("net_mod", str(os.path.join(net_path, net_name + '.py')))
            (train_data, train_labels), (test_data, test_labels) = net_mod.get_formatted_data(os.path.join(temp_path, temp_name + ".hf5"))
        except IOError:
            raise NameError('There is no function named "get_formatted_data" in %s.py.' % (net_name))
            return()
    else:
        (train_data, train_labels), (test_data, test_labels) = load_data(os.path.join(temp_path, temp_name + ".hf5"))
    
    
    #Check sizes of loaded data against the input and output shape of the net
    #train_data, test_data, ignored_error_reshaping = reshape_data(train_data, test_data, input_layer_shape, **opt_arg)
    
    #if ignored_error_reshaping:
        #ignored_error = True
    
    #train_labels, test_labels, ignored_error_reshaping = reshape_data(train_labels, test_labels, output_layer_shape, **opt_arg)
    
    #if ignored_error_reshaping:
        #ignored_error = True
    
    #Training takes place here
    if not opt_arg['only_print_image']:
        wiki_data['training'] = {}
        wiki_data['training']['time_start'] = time.gmtime(time.time())
        
        hist = _train_net(net, net_name, train_data, test_data, train_labels, test_labels, **opt_arg)
        
        wiki_data['training']['time_end'] = time.gmtime(time.time())
        
    #Give a warning if some errors were ignored and not specifically said
    #(i.e. user input) to be automatically handeled
    if ignored_error:
        print(bcolors.WARNING + "This run ignored errors along the way!" + bcolors.ENDC)
    
    #Plot the distribution of labels against predictions
    train_calculated_snr, test_calculated_snr = load_calculated_snr(os.path.join(temp_path, temp_name + ".hf5"))
    #print("Training calculated snr: {}".format(train_calculated_snr))
    #print("Testing calculated snr: {}".format(test_calculated_snr))
    t_string = date_to_file_string(wiki_data['training']['time_start'])
    wiki_data['SNR_plot_name'] = net_name + '_snr_' + t_string + '.png'
    plot_true_and_calc(net, test_data, test_labels, test_calculated_snr, os.path.join(net_path, wiki_data['SNR_plot_name']), show=opt_arg['show_snr_plot'], net_name=net_name)
    
    #Plot the loss over some recorded history
    try:
        wiki_data['loss_plot_name'] = net_name + '_loss_plot_' + t_string + '.png'
        make_loss_plot(os.path.join(get_store_path(), net_name + "_results.json"), os.path.join(get_store_path(), wiki_data['loss_plot_name']))
    except IOError:
        print(bcolors.OKGREEN + 'Could not create plot of the loss function, as the %s file could not be found.' % (net_name + '_results.json') + bcolors.ENDC)
    
    
    #Store wiki data about the loss
    try:
        wiki_data['loss'] = read_json(os.path.join(get_store_path(), net_name + "_results.json"))
    except IOError:
        losses = hist.history['loss']
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
    wiki_data['ignored_errors'] = ignored_error
    wiki_data['template_properties'] = load_parameter_space(os.path.join(temp_path, temp_name + ".hf5"))
    wiki_data['network'] = model_to_string(net)
    
    #Create a wiki-entry
    if opt_arg['create_wiki_entry']:
        make_wiki_entry(wiki_data)
