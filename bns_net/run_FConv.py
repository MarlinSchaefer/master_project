from tensorflow import ConfigProto, Session
config = ConfigProto()
config.gpu_options.allow_growth = True
session = Session(config=config)
from run_net import run_net
import numpy as np
import traceback

if __name__ == "__main__":    
    try:
        msg = 'ATTENTION: CHANGED THE DEFAULT IMPORT SIZE IN D_OBJ! Using custom loss for WConv1D. (metric mape)'
        run_net('FConv1D', 'templates_new_vary_mass_sky_pos_coa_phase_large', ini_file='testing_net.ini', create_wiki_entry=True, overwrite_template_file=False, epochs=20, use_data_object=True, show_snr_plot=False, overwrite_net_file=True, evaluate_on_large_testing_set=False, batch_size=24, custom_message=msg)
        #run_net('FConv1D', 'templates_new', ini_file='testing_net.ini', create_wiki_entry=True, overwrite_template_file=False, epochs=20, use_data_object=True, show_snr_plot=False, overwrite_net_file=True, evaluate_on_large_testing_set=False, batch_size=24, custom_message=msg)
    except:
        traceback.print_exc()
        pass
