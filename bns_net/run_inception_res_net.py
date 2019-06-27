from tensorflow import ConfigProto, Session
config = ConfigProto()
config.gpu_options.allow_growth = True
session = Session(config=config)
from run_net import run_net
import numpy as np
import traceback

if __name__ == "__main__":
    try:
        msg = 'This network has increasing numbers of filters in all inception layers. Dimensional reduction is applied to reshape the outputs of most inception layers, to be able to build residual connections. (Trained using mape, metric mse)'
        run_net('inception_res_net_rev_6', 'templates_new_dev24', ini_file='testing_net.ini', create_wiki_entry=True, overwrite_template_file=False, epochs=50, use_data_object=True, show_snr_plot=False, overwrite_net_file=True, evaluate_on_large_testing_set=False, batch_size=24, custom_message=msg)
    except:
        traceback.print_exc()
        pass
