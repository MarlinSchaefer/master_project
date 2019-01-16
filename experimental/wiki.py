import os
import json
import numpy as np
import time

def read_json(path):
    with open(path, 'r') as FILE:
        data = json.load(FILE)
    
    min_tr = (0, np.inf)
    min_te = (0, np.inf)
    
    for pt in data:
        if pt[1][0] < min_tr[1]:
            min_tr = (pt[0], pt[1][0])
        if pt[2][0] < min_te[1]:
            min_te = (pt[0], pt[1][0])
    
    return({'min_training': min_tr, 'min_testing': min_te, 'last_training': (data[-1][0], data[-1][1][0]), 'last_testing': (data[-1][0], data[-1][2][0])})

def time_to_string(t):
    return("{}.{}.{}: {}:{}.{}".format(t.tm_mday, t.tm_mon, t.tm_year, t.tm_hour, t.tm_min, t.tm_sec))

#Maybe implement this to just give the model creation
def net_to_wiki(file_path):
    return

def get_wiki_path():
    return(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wiki"))

def model_to_string(model):
    summary = []
    model.summary(print_fn=lambda x: summary.append(x + '\n'))
    
    s = '\n'
    
    for st in summary:
        s += st
        
    return(s)

def export_to_wiki_file(data):
    file_path = os.path.join(get_wiki_path(), 'wiki.txt')
    file_already_exists = os.path.isfile(file_path)
    
    with open(file_path, 'a+') as FILE:
        if file_already_exists:
            FILE.write('###############################################################################')
            FILE.write('\n\n\n')
        
        for s in data:
            FILE.write(s)
        
    return()

def make_wiki_entry(data):
    order = ['training', 'loss', 'network', 'loss', 'template_properties', 'programm_internals', 'template_generation']
    
    output = []
    
    for key in order:
        if type(data[key]) == dict:
            #Do stuff
            output.append(str(key) + ':')
            output.append('\n')
            
            for k, v in data[key].items():
                if isinstance(v, type(time.gmtime(time.time()))):
                    output.append('\t' + str(k) + ': ' + time_to_string(v) + '\n')
                else:
                    output.append('\t' + str(k) + ': ' + str(v) + '\n')
            
            output.append('\n')
        else:
            #Do other stuff
            if isinstance(data[key], type(time.gmtime(time.time()))):
                output.append(str(key) + ': ' + time_to_string(data[key]) + '\n')
            else:
                output.append(str(key) + ': ' + str(data[key]) + '\n')
            
            output.append('\n')
    
    #Handle all other keys that are not handled before
    for key, val in data.items():
        if not key in order:
            if type(data[key]) == dict:
                #Do stuff
                output.append(str(key) + ':')
                output.append('\n')
                
                for k, v in data[key].items():
                    if isinstance(v, type(time.gmtime(time.time()))):
                        output.append('\t' + str(k) + ': ' + time_to_string(v) + '\n')
                    else:
                        output.append('\t' + str(k) + ': ' + str(v) + '\n')
                
                output.append('\n')
            else:
                #Do other stuff
                if isinstance(data[key], type(time.gmtime(time.time()))):
                    output.append(str(key) + ': ' + time_to_string(data[key]) + '\n')
                else:
                    output.append(str(key) + ': ' + str(data[key]) + '\n')
                
                output.append('\n')
    
    export_to_wiki_file(output)
