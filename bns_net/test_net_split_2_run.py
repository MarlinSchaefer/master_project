from tensorflow import ConfigProto, Session
config = ConfigProto()
config.gpu_options.allow_growth = True
session = Session(config=config)
from run_net import run_net
import numpy as np

ep = 150
wiki_e = True

if __name__ == "__main__":
    try:
        msg = 'This network is basically the same as collect_inception_net_6_rev_5, but decreases the initial learning rate for the Adam optimizer by a factor of two.'
        run_net('collect_inception_net_6_rev_6', 'templates_new', ini_file='testing_net.ini', create_wiki_entry=True, overwrite_template_file=False, epochs=100, use_data_object=True, show_snr_plot=False, overwrite_net_file=True, evaluate_on_large_testing_set=False, batch_size=24, custom_message=msg)
    except:
        pass
    msg = 'This network is basically the same as collect_inception_net_6_rev_6, but introduces a residual connection after every inception layer. This means, that the input of every layer gets added back to the output. This is supposed to help during training.'
    run_net('collect_inception_net_6_rev_7', 'templates_new', ini_file='testing_net.ini', create_wiki_entry=True, overwrite_template_file=False, epochs=100, use_data_object=True, show_snr_plot=False, overwrite_net_file=True, evaluate_on_large_testing_set=False, batch_size=24, custom_message=msg)
    
