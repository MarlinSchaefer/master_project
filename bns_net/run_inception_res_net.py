from tensorflow import ConfigProto, Session
config = ConfigProto()
config.gpu_options.allow_growth = True
session = Session(config=config)
from run_net import run_net
import numpy as np
import traceback

if __name__ == "__main__":
    try:
        msg = 'ATTENTION: CHANGED THE DEFAULT IMPORT SIZE IN D_OBJ! This run is supposed to cross-check the first "final" run, i.e. tcn_collect_inception_res_net_rev_6. It uses the same data, but less and also checkpoints every epoch. It removes auxilliary outputs and introduces two further inception layers after the final concatenation.'
        run_net('tcn_collect_inception_res_net_rev_7', 'templates_more_right_vary_all_final', ini_file='testing_net.ini', create_wiki_entry=True, overwrite_template_file=False, epochs=20, use_data_object=True, show_snr_plot=False, overwrite_net_file=True, evaluate_on_large_testing_set=False, batch_size=16, custom_message=msg)
    except:
        traceback.print_exc()
        pass
    
    #try:
        #msg = 'ATTENTION: CHANGED THE DEFAULT IMPORT SIZE IN D_OBJ! This network is collect_inception_res_net_rev_5 from the prior run, but uses the fixed custom loss function. (loss_c1)'
        #run_net('collect_inception_res_net_rev_5', 'templates_new_vary_mass_sky_pos_coa_phase_large', ini_file='testing_net.ini', create_wiki_entry=True, overwrite_template_file=False, epochs=40, use_data_object=True, show_snr_plot=False, overwrite_net_file=True, evaluate_on_large_testing_set=False, batch_size=24, custom_message=msg)
        ##run_net('collect_inception_res_net_rev_5', 'templates_new', ini_file='testing_net.ini', create_wiki_entry=True, overwrite_template_file=False, epochs=40, use_data_object=True, show_snr_plot=False, overwrite_net_file=True, evaluate_on_large_testing_set=False, batch_size=24, custom_message=msg)
    #except:
        #traceback.print_exc()
        #pass
