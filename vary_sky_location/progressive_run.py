from run_net import run_net
import numpy as np

"""
NUM_TEMP = 1000

if __name__ == "__main__":
    run_net('progressive_net', 'progressive_data_9', epochs=50, random_starting_time=True, coa_phase=[0.0, 2 * np.pi], snr=[90.0,100.0], mass2=[10.0,30.0], mass1=[10.0,30.0], num_of_templates=NUM_TEMP, overwrite_template_file=True, ignore_fixable_errors=False, only_print_image=False, use_custom_train_function=True, epoch_break=5, declination=[-np.pi/2, np.pi/2], right_ascension=[-np.pi, np.pi], polarization=[0.0, np.pi], show_snr_plot=False, overwrite_net_file=True)
    
    it = range(2, 9)
    it.reverse()
    
    for i in it:
        print("Currently in i = {}".format(i))
        run_net('progressive_net', 'progressive_data_' + str(i), epochs=50, random_starting_time=True, coa_phase=[0.0, 2 * np.pi], snr=[i*10.0, (i+1)*10.0], mass2=[10.0,30.0], mass1=[10.0,30.0], num_of_templates=NUM_TEMP, overwrite_template_file=True, ignore_fixable_errors=False, only_print_image=False, use_custom_train_function=True, epoch_break=5, declination=[-np.pi/2, np.pi/2], right_ascension=[-np.pi, np.pi], polarization=[0.0, np.pi], show_snr_plot=False, overwrite_net_file=False)
    
    run_net('progressive_net', 'progressive_data_0', epochs=50, random_starting_time=True, coa_phase=[0.0, 2 * np.pi], snr=[6.0, 15.0], mass2=[10.0,30.0], mass1=[10.0,30.0], num_of_templates=10*NUM_TEMP, overwrite_template_file=True, ignore_fixable_errors=False, only_print_image=False, use_custom_train_function=True, epoch_break=5, declination=[-np.pi/2, np.pi/2], right_ascension=[-np.pi, np.pi], polarization=[0.0, np.pi], show_snr_plot=True, overwrite_net_file=False)
"""

if __name__ == "__main__":
    run_net('progressive_net', 'progressive_data_9', ini_file='progressive_net.ini', overwrite_net_file=True, snr=[90.0, 100.0])
    
    it = range(2, 9)
    it.reverse()
    
    for i in it:
        print("Currently in i = {}".format(i))
        run_net('progressive_net', 'progressive_data_' + str(i), ini_file='progressive_net.ini', snr=[i*10.0, (i+1)*10.0])
    
    run_net('progressive_net', 'progressive_data_0', ini_file='progressive_net.ini', epochs=150, snr=[6.0, 15.0], num_of_templates=10000, show_snr_plot=True)
