import os
import keras
from load_data import load_data
import numpy as np
import imp
from make_snr_plot import plot

"""
TODO:
-Implement the option to not overwrite existing files
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
"""
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
    opt_arg['overwrite_template_file'] = False
    opt_arg['overwrite_net_file'] = True
    opt_arg['show_snr_plot'] = True
    opt_arg['only_print_image'] = False
    opt_arg['use_custom_train_function'] = False
    
    for key in opt_arg.keys():
        if key in kwargs:
            opt_arg[key] = kwargs.get(key)
            del kwargs[key]
    
    net_path = opt_arg['net_path']
    temp_path = opt_arg['temp_path']
    
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
    
    input_layer_shape = (net.layers)[0].get_input_at(0).get_shape().as_list()
    input_layer_shape = tuple(input_layer_shape[1:])
    output_layer_shape = (net.layers)[-1].get_output_at(0).get_shape().as_list()
    output_layer_shape = tuple(output_layer_shape[1:])
    
    #Handle not existing template file
    #Either create or quit
    if not template_exists_q(temp_name, path=temp_path) or opt_arg['overwrite_template_file']:
        if not opt_arg['ignore_fixable_errors'] and not opt_arg['overwrite_template_file']:
            inp = raw_input('No template file named %s found at %s.\nDo you want to generate it?\n' % (temp_name, temp_path))
        else:
            inp = 'y'
            if not opt_arg['overwrite_template_file']:
                ignored_error = True
        
        if input_to_bool(inp):
            
            kwargs['data_shape'] = input_layer_shape
            kwargs['label_shape'] = output_layer_shape
            if 'temp_creation_script' in kwargs:
                temp_creation_script = kwargs.get('temp_creation_script')
                del kwargs['temp_creation_script']
                try:
                    #The custom module needs to have a 'create_file' method
                    custom_temp_script = importlib.import_module(str(temp_creation_script))
                    
                    custom_temp_script.create_file(name=temp_name, path=temp_path, **kwargs)
                except ImportError:
                    print("Could not import the creation file.")
                    return()
            else:
                try:
                    from make_template_bank_new import create_file
                    #from make_template_bank import create_file
                    
                    create_file(name=temp_name, path=temp_path, **kwargs)
                except ImportError:
                    print("Could not import module 'make_template_file'")
                    return()
        else:
            return()
    
    #Load templates
    print(os.path.join(temp_path, temp_name + ".hf5"))
    (train_data, train_labels), (test_data, test_labels) = load_data(os.path.join(temp_path, temp_name + ".hf5"))
    
    #Check sizes of loaded data against the input and output shape of the net
    do_reshape = False
    print('Output shape: {}'.format(output_layer_shape))
    print('data shape: {}'.format(train_labels.shape))
    if not train_data[0].shape == input_layer_shape:
        if not opt_arg['ignore_fixable_errors']:
            inp = 'False'
            inp = raw_input("The provided data does not fit the provided net.\nDo you want to try and reshape the data?\n")
            if input_to_bool(inp):
                do_reshape = True
            else:
                return
        else:
            do_reshape = True
            ignored_error = True
    
    if do_reshape:
        cache = list(input_layer_shape)
        cache.insert(0, len(train_data))
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
                return
        else:
            do_reshape = True
            ignored_error = True
    
    if do_reshape:
        cache = list(output_layer_shape)
        cache.insert(0, len(train_labels))
        np.reshape(train_labels, tuple(cache))
        
        cache[0] = len(test_labels)
        np.reshape(test_labels, tuple(cache))
    
    if not opt_arg['only_print_image']:
        #If everything is fine, train and evaluate the net
        net.compile(loss=opt_arg['loss'], optimizer=opt_arg['optimizer'], metrics=opt_arg['metrics'])
        
        print(net.summary())
        if opt_arg['use_custom_train_function']:
            try:
                #NOTE: The module needs to have a method 'train_model', which returns the trained model.
                net_mod = imp.load_source("net_mod", str(os.path.join(net_path, net_name + '.py')))
                net = net_mod.train_model(net)
            
            except IOError:
                raise NameError('There is no net named %s in %s.' % (net_name, net_path))
                return()
        else:
            net.fit(train_data, train_labels, epochs=opt_arg['epochs'])
        
        net.save(os.path.join(net_path, net_name + '.hf5'))
        
        print(net.evaluate(test_data, test_labels))
    
    if ignored_error:
        print(bcolors.WARNING + "This run ignored errors along the way!" + bcolors.ENDC)
    
    plot(net, test_data, test_labels, os.path.join(net_path, net_name + '_snr.png'), show=opt_arg['show_snr_plot'], net_name=net_name)
