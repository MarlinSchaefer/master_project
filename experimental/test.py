from run_net import run_net
import numpy as np

if __name__ == "__main__":
    run_net('varied_nothing_net_v2', 'test_2', epochs=10, coa_phase=[0.0, np.pi],random_starting_time=True, snr=[5.0,12.0], num_of_templates=20,overwrite_template_file=True,ignore_fixable_errors=False,only_print_image=False)
