from run_net import run_net
import numpy as np

if __name__ == "__main__":
    run_net('test_net_split_2', 'mult_output_data_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=True, overwrite_template_file=False, epochs=10, num_of_templates=100000, format_data=True, epoch_break=5, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=500)
