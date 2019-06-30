from tensorflow import ConfigProto, Session
config = ConfigProto()
config.gpu_options.allow_growth = True
session = Session(config=config)
from run_net import run_net
import numpy as np
import traceback

if __name__ == "__main__":
    try:
        msg = 'This network tries to combine the sensitivity of inception_res_net_rev_9 with the recovery of SNR-values of collect_inception_net_rev_(x). The architecture is therefore a combination of the two architectures, utilizing what has been identified as best practice. (i.e. having residual connections, using kernel sizes (1, 2, 3), having convolution layers before the inception stack, using the same layers after concatination...)(Trained using mse, metric mape)'
        run_net('collect_inception_res_net_rev_1', 'templates_new_vary_mass_sky_pos_cos_phase', ini_file='testing_net.ini', create_wiki_entry=True, overwrite_template_file=False, epochs=50, use_data_object=True, show_snr_plot=False, overwrite_net_file=True, evaluate_on_large_testing_set=False, batch_size=24, custom_message=msg)
    except:
        traceback.print_exc()
        pass
