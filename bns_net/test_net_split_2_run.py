from tensorflow import ConfigProto, Session
config = ConfigProto()
config.gpu_options.allow_growth = True
session = Session(config=config)
from run_net import run_net
import numpy as np

ep = 150
wiki_e = True

if __name__ == "__main__":
    msg = 'Trying to use the new way of generating training data, i.e. having signals and noise separated. Now also the learning rate decays (with a rate of 0.1), in order to combat numerical instabilities.'
    run_net('collect_inception_net_6_rev_6', 'templates_new', ini_file='testing_net.ini', create_wiki_entry=True, overwrite_template_file=False, epochs=100, use_data_object=True, show_snr_plot=False, overwrite_net_file=True, evaluate_on_large_testing_set=False, batch_size=24, custom_message=msg)
    
