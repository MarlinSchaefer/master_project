from run_net import run_net
import numpy as np

if __name__ == "__main__":
    run_net('inception_net', 'mult_output_data', ini_file='inception_net.ini', snr=[10.0, 50.0], create_wiki_entry=True, overwrite_template_file=True, epochs=75, num_of_templates=100, format_data=True, epoch_break=3, gw_prob=0.5, use_custom_compilation=True, overwrite_net_file=True, t_from_right=2.5, show_snr_plot=False)
