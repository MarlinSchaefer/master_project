from run_net import run_net
import numpy as np

if __name__ == "__main__":
    run_net('longterm_net', 'test', epochs=150, random_starting_time=True, coa_phase=[0.0, 2 * np.pi], snr=[6.0,15.0], mass2=[10.0,30.0], mass1=[10.0,30.0], num_of_templates=10000, overwrite_template_file=False, ignore_fixable_errors=False, only_print_image=False, use_custom_train_function=True, epoch_break=5, declination=[-np.pi/2, np.pi/2], right_ascension=[-np.pi, np.pi], polarization=[0.0, np.pi])
