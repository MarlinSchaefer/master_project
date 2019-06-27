from tensorflow import ConfigProto, Session
config = ConfigProto()
config.gpu_options.allow_growth = True
session = Session(config=config)
from run_net import run_net
import numpy as np
import traceback

if __name__ == "__main__":
    try:
        msg = 'This network is the one inception_res_net_rev_5, but adapts the filter sizes to 4, 8, 16. (Trained using mape, metric mse)'
        run_net('inception_res_net_rev_5', 'templates_new_dev24', ini_file='testing_net.ini', create_wiki_entry=True, overwrite_template_file=False, epochs=50, use_data_object=True, show_snr_plot=False, overwrite_net_file=True, evaluate_on_large_testing_set=False, batch_size=24, custom_message=msg)
    except:
        traceback.print_exc()
        pass
    
    try:
        msg = 'This network is the one inception_res_net_rev_2 should have been. It fixes some wrong connections in the network. And fixes a bug, where the signals were overwritten by noise. Also the filter sizes were changed to 4, 8, 16. (Trained using mse, metric mape)'
        run_net('inception_res_net_rev_7', 'templates_new_dev24', ini_file='testing_net.ini', create_wiki_entry=True, overwrite_template_file=False, epochs=50, use_data_object=True, show_snr_plot=False, overwrite_net_file=True, evaluate_on_large_testing_set=False, batch_size=24, custom_message=msg)
    except:
        traceback.print_exc()
        pass
    
    try:
        msg = 'This network is the one inception_res_net_rev_4 should have been. It fixes some wrong connections in the network. And fixes a bug, where the signals were overwritten by noise. Also the filter sizes were changed to 4, 8, 16. (Trained using mape, metric mse)'
        run_net('inception_res_net_rev_8', 'templates_new_dev24', ini_file='testing_net.ini', create_wiki_entry=True, overwrite_template_file=False, epochs=50, use_data_object=True, show_snr_plot=False, overwrite_net_file=True, evaluate_on_large_testing_set=False, batch_size=24, custom_message=msg)
    except:
        traceback.print_exc()
        pass
