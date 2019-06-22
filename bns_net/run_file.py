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
        msg = 'This network is  the same as collect_inception_net_3_rev_7. The difference comes from the generator, that has different signals and noise instanciations for training and testing.'
        run_net('collect_inception_net_3_rev_8', 'templates_new', ini_file='testing_net.ini', create_wiki_entry=True, overwrite_template_file=False, epochs=100, use_data_object=True, show_snr_plot=False, overwrite_net_file=True, evaluate_on_large_testing_set=False, batch_size=24, custom_message=msg)
    except:
        traceback.print_exc()
        pass
