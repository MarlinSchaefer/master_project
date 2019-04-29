import ConfigParser
import os
import ast

def _section_to_dict(config, sec):
    options = config.options(sec)
    
    ret = {}
    
    for option in options:
        ret[option] = ast.literal_eval(config.get(sec, option))
    
    return(ret)
    

def load_options(file_name):
    config = ConfigParser.ConfigParser()
    
    config.read(file_name)
    
    sec = config.sections()
    
    ret = {}
    
    for s in sec:
        dic = _section_to_dict(config, s)
        ret.update(dic)
    
    return(ret)

def run_net_defaults():
    config = ConfigParser.ConfigParser()
    
    config.read('defaults.ini')
    
    return(_section_to_dict(config, 'run_net'))

def make_template_bank_defaults():
    config = ConfigParser.ConfigParser()
    
    config.read('defaults.ini')
    
    return((_section_to_dict(config, 'make_template_bank_wav'), _section_to_dict(config, 'make_template_bank_opt')))

def evaluate_net_defaults():
    config = ConfigParser.ConfigParser()
    
    config.read('defaults.ini')
    
    return(_section_to_dict(config, 'evaluate_nets'))
