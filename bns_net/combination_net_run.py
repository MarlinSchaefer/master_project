from run_net import run_net
from aux_functions import get_store_path
import numpy as np
import os
from generate_split_data import generate_template

ep = 150
wiki_e = True
file_name = 'templates'
num_signals = 20000
num_noise = 100000

generate_template(os.path.join(get_store_path(), file_name + '.hf5'), num_signals, num_noise, snr=[8.0, 15.0])

if __name__ == "__main__":
    run_net('combination_net', file_name, ini_file='testing_net.ini', create_wiki_entry=False, overwrite_template_file=False, epochs=1, use_data_object=True, show_snr_plot=False, overwrite_net_file=True, evaluate_on_large_testing_set=False)
    
