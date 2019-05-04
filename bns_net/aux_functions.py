import os

def date_to_file_string(t):
    return("{}{}{}{}{}{}".format(t.tm_mday, t.tm_mon, t.tm_year, t.tm_hour, t.tm_min, t.tm_sec))

def get_store_path():
    return(os.path.join(os.path.dirname(os.path.abspath(__file__)), "saves"))

def filter_keys(opt_arg, kwargs):
    for key in opt_arg.keys():
        if key in kwargs:
            opt_arg[key] = kwargs.get(key)
            del kwargs[key]
    
    return(opt_arg, kwargs)
