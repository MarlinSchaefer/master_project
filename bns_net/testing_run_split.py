from run_net import run_net
import numpy as np

if __name__ == "__main__":
    run_net('testing_net_split', 'mult_output_data', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=True, overwrite_template_file=False, epochs=150, num_of_templates=10000, format_data=True, epoch_break=2, gw_prob=0.5, use_custom_compilation=True)
