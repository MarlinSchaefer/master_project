from run_net import run_net
import numpy as np

if __name__ == "__main__":
    run_net('longterm_net', 'test', epochs=30, coa_phase=[0.0, np.pi],random_starting_time=True, snr=[1.0,12.0], num_of_templates=10000,overwrite_template_file=True,ignore_fixable_errors=True,only_print_image=False, use_custom_train_function=True, epoch_break=10)
