from tensorflow import ConfigProto, Session
config = ConfigProto()
config.gpu_options.allow_growth = True
session = Session(config=config)
from run_net import run_net
import numpy as np
import traceback

if __name__ == "__main__":
    try:
        msg = 'This network is basically the same as inception_res_net, but uses different data to train. Now the segments don\'t overlap which causes the numer of input samples to be almost halved. (Input samples means points per input and not samples the network is trained on.) The idea was inspired by Christoph Dreißigacker on a conversation on 26th of June 2019 in our office.'
        run_net('inception_res_net_rev_2', 'templates_new', ini_file='testing_net.ini', create_wiki_entry=True, overwrite_template_file=False, epochs=50, use_data_object=True, show_snr_plot=False, overwrite_net_file=True, evaluate_on_large_testing_set=False, batch_size=24, custom_message=msg)
    except:
        traceback.print_exc()
        pass
    
    try:
        msg = 'This network is the same as inception_res_net, but uses mean average percentage error to train the SNR-part. It seems, that this statistic is correlated to the sensitivity more strongly than mean squared error.'
        run_net('inception_res_net_rev_3', 'templates_new', ini_file='testing_net.ini', create_wiki_entry=True, overwrite_template_file=False, epochs=50, use_data_object=True, show_snr_plot=False, overwrite_net_file=True, evaluate_on_large_testing_set=False, batch_size=24, custom_message=msg)
    except:
        traceback.print_exc()
        pass
    
    try:
        msg = 'This network is the same as inception_res_net_2, but uses mean average percentage error to train the SNR-part. It seems, that this statistic is correlated to the sensitivity more strongly than mean squared error.'
        run_net('inception_res_net_rev_3', 'templates_new', ini_file='testing_net.ini', create_wiki_entry=True, overwrite_template_file=False, epochs=50, use_data_object=True, show_snr_plot=False, overwrite_net_file=True, evaluate_on_large_testing_set=False, batch_size=24, custom_message=msg)
    except:
        traceback.print_exc()
        pass
