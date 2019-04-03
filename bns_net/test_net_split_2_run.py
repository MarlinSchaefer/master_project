from run_net import run_net
import numpy as np

if __name__ == "__main__":
    run_net('inception_net', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=True, overwrite_template_file=False, epochs=100, num_of_templates=100000, format_data=True, epoch_break=5, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32)
