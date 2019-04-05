from run_net import run_net
import numpy as np

ep = 100
wiki_e = True

if __name__ == "__main__":
    #run_net('inception_net_2', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=True, overwrite_template_file=False, epochs=ep, num_of_templates=20000, format_data=True, epoch_break=5, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32)
    
    #run_net('inception_net_4', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=True, overwrite_template_file=False, epochs=ep, num_of_templates=20000, format_data=True, epoch_break=5, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32)
    
    #run_net('inception_net_6', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=True, overwrite_template_file=False, epochs=ep, num_of_templates=20000, format_data=True, epoch_break=5, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32)
    
    #run_net('inception_net_8', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=True, overwrite_template_file=False, epochs=ep, num_of_templates=20000, format_data=True, epoch_break=5, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32)
    
    try:
        run_net('inception_net_10', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=wiki_e, overwrite_template_file=False, epochs=1000, num_of_templates=20000, format_data=True, epoch_break=5, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32)
    except:
        pass
    
    try:
        run_net('inception_net_6_skip', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=wiki_e, overwrite_template_file=False, epochs=1000, num_of_templates=20000, format_data=True, epoch_break=5, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32, custom_message='This net takes as input the channels 2048Hz, 512Hz, 128Hz')
    except:
        pass
    
    try:
        run_net('inception_net_2_small', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=wiki_e, overwrite_template_file=False, epochs=ep, num_of_templates=20000, format_data=True, epoch_break=5, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32)
    except:
        pass
    
    try:
        run_net('inception_net_4_small', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=wiki_e, overwrite_template_file=False, epochs=ep, num_of_templates=20000, format_data=True, epoch_break=5, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32)
    except:
        pass
    
    try:
        run_net('inception_net_6_small', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=wiki_e, overwrite_template_file=False, epochs=ep, num_of_templates=20000, format_data=True, epoch_break=5, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32)
    except:
        pass
    
    try:
        run_net('inception_net_8_small', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=wiki_e, overwrite_template_file=False, epochs=ep, num_of_templates=20000, format_data=True, epoch_break=5, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32)
    except:
        pass
    
    try:
        run_net('inception_net_10_small', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=wiki_e, overwrite_template_file=False, epochs=ep, num_of_templates=20000, format_data=True, epoch_break=5, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32)
    except:
        pass
    
    try:
        run_net('inception_net_12_small', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=wiki_e, overwrite_template_file=False, epochs=ep, num_of_templates=20000, format_data=True, epoch_break=5, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32)
    except:
        pass
    
