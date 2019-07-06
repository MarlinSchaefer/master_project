from tensorflow import ConfigProto, Session
config = ConfigProto()
config.gpu_options.allow_growth = True
session = Session(config=config)
from run_net import run_net
import numpy as np
import traceback

if __name__ == "__main__":
    #try:
        #msg = 'This run tries to identify which data is best to be fed to the network, as there is the new and reduced data feeding. In this case we try to have just non overlapping data for all sample rates. (Trained using mse, metric mape)'
        #run_net('inception_res_net_rev_9', 'templates_new_vary_mass_sky_pos_coa_phase', ini_file='testing_net.ini', create_wiki_entry=True, overwrite_template_file=False, epochs=50, use_data_object=True, show_snr_plot=False, overwrite_net_file=True, evaluate_on_large_testing_set=False, batch_size=24, custom_message=msg)
    #except:
        #traceback.print_exc()
        #pass
    
    #try:
        #msg = 'This run tries to identify which data is best to be fed to the network, as there is the new and reduced data feeding. In this case we try to have non overlapping data for only the sample rates 2048, 512 and 128. (Trained using mse, metric mape)'
        #run_net('inception_res_net_rev_11', 'templates_new_vary_mass_sky_pos_coa_phase', ini_file='testing_net.ini', create_wiki_entry=True, overwrite_template_file=False, epochs=50, use_data_object=True, show_snr_plot=False, overwrite_net_file=True, evaluate_on_large_testing_set=False, batch_size=24, custom_message=msg)
    #except:
        #traceback.print_exc()
        #pass
    
    try:
        msg = 'ATTENTION: CHANGED THE DEFAULT IMPORT SIZE IN D_OBJ! Removed dropout from inception_res_net_rev_9 and increased the number of signals and noise samples to 250000 and 750000 respectively. (Trained using mse, metric mape)'
        run_net('inception_res_net_rev_15', 'templates_new_vary_mass_sky_pos_coa_phase', ini_file='testing_net.ini', create_wiki_entry=True, overwrite_template_file=False, epochs=20, use_data_object=True, show_snr_plot=False, overwrite_net_file=True, evaluate_on_large_testing_set=False, batch_size=24, custom_message=msg)
    except:
        traceback.print_exc()
        pass
    
    try:
        msg = 'ATTENTION: CHANGED THE DEFAULT IMPORT SIZE IN D_OBJ! Removed dropout from collect_inception_res_net_rev_1 and increased the number of signals and noise samples to 250000 and 750000 respectively. (Trained using mse, metric mape)'
        run_net('collect_inception_res_net_rev_2', 'templates_new_vary_mass_sky_pos_coa_phase', ini_file='testing_net.ini', create_wiki_entry=True, overwrite_template_file=False, epochs=20, use_data_object=True, show_snr_plot=False, overwrite_net_file=True, evaluate_on_large_testing_set=False, batch_size=24, custom_message=msg)
    except:
        traceback.print_exc()
        pass
