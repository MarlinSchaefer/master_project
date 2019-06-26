from tensorflow import ConfigProto, Session
config = ConfigProto()
config.gpu_options.allow_growth = True
session = Session(config=config)
from run_net import run_net, get_store_path
import numpy as np
import traceback
import imp
import os

if __name__ == "__main__":
    hyper_explore = imp.load_source('hyper_explore', os.path.join(get_store_path(), 'hyper_param_explore.py'))
    possibilities = [(4, 16, 20), (1, 16, 20), (3, 4, 8), (8, 16, 32), 
                     (8, 10, 64), (6, 12, 30), (2, 4, 8), (5, 10, 36), 
                     (6, 8, 12), (8, 16, 20), (4, 6, 8), (8, 10, 12), 
                     (1, 2, 4), (4, 16, 64), (1, 16, 64), (4, 8, 16), 
                     (1, 2, 64), (8, 16, 64), (5, 6, 8), (4, 8, 32)]
    for poss in possibilities:
        try:
            hyper_explore.set_filter_size(poss)
            msg = 'Trying the filter sizes: {}'.format(poss)
            run_net('hyper_param_explore', 'templates_new_large', ini_file='testing_net.ini', create_wiki_entry=True, overwrite_template_file=False, epochs=20, use_data_object=True, show_snr_plot=False, overwrite_net_file=True, evaluate_on_large_testing_set=False, batch_size=24, custom_message=msg)
        except:
            traceback.print_exc()
            pass
