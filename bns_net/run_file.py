from tensorflow import ConfigProto, Session
config = ConfigProto()
config.gpu_options.allow_growth = True
session = Session(config=config)
from run_net import run_net
import numpy as np
import traceback

ep = 150
wiki_e = True

if __name__ == "__main__":
    try:
        msg = 'This network is basically the same as collect_inception_net_6_rev_6, but with less channels, to consume less GPU-memory. This is the ground truth to compare all modifications to. (i.e. compare the accuracy of this against collect_inception_net_6_rev_6 and put into relation all following results using this architecture.)'
        run_net('collect_inception_net_3_rev_6', 'templates_new', ini_file='testing_net.ini', create_wiki_entry=True, overwrite_template_file=False, epochs=100, use_data_object=True, show_snr_plot=False, overwrite_net_file=True, evaluate_on_large_testing_set=False, batch_size=24, custom_message=msg)
    except:
        traceback.print_exc()
        pass
