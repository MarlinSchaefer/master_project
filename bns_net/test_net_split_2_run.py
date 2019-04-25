from run_net import run_net
import numpy as np

ep = 150
wiki_e = True

if __name__ == "__main__":
    run_net('test_dobj_net', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=True, overwrite_template_file=False, epochs=100, num_of_templates=200, gw_prob=0.5, use_data_object=True, show_snr_plot=False, custom_message='First longterm test of the new metric functions.')
    #try:
        #run_net('collect_inception_net_6_rev_6', 'small_snr_7_100000', ini_file='testing_net.ini', snr=[8.0, 15.0], create_wiki_entry=wiki_e, overwrite_template_file=True, epochs=0, num_of_templates=100000, gw_prob=0.5, custom_message='Generated large template file at more realistic SNRs.')
    #except:
        #pass
    
    #try:
        #run_net('collect_inception_net_6_rev_6', 'small_snr_vary_sky_7_100000', ini_file='testing_net.ini', snr=[8.0, 15.0], create_wiki_entry=wiki_e, overwrite_template_file=True, epochs=0, num_of_templates=100000, gw_prob=0.5, right_ascension=[-np.pi, np.pi], declination=[-np.pi/2, np.pi/2], polarization=[0.0, np.pi], custom_message='Generated large template file at more realistic SNRs with varied sky position.')
    #except:
        #pass
    
    #try:
        #run_net('collect_inception_net_6_rev_6', 'small_snr_vary_sky_masses_7_100000', ini_file='testing_net.ini', snr=[8.0, 15.0], create_wiki_entry=wiki_e, overwrite_template_file=True, epochs=0, num_of_templates=100000, gw_prob=0.5, right_ascension=[-np.pi, np.pi], declination=[-np.pi/2, np.pi/2], polarization=[0.0, np.pi], mass1=[1.3, 1.5], mass2=[1.3, 1.5], custom_message='Generated large template file at more realistic SNRs with varied sky position and varied masses.')
    #except:
        #pass
    
    #try:
        #run_net('collect_inception_net_6_rev_6', 'small_snr_3_100000', ini_file='testing_net.ini', snr=[8.0, 15.0], create_wiki_entry=wiki_e, overwrite_template_file=True, epochs=0, num_of_templates=100000, gw_prob=0.5, resample_delta_t=(1.0/2048, 1.0/512, 1.0/128), resample_t_len=(2.0, 8.0, 32.0), custom_message='Generated small template file at more realistic SNRs.')
    #except:
        #pass
    
    #try:
        #run_net('collect_inception_net_6_rev_6', 'small_snr_vary_sky_3_100000', ini_file='testing_net.ini', snr=[8.0, 15.0], create_wiki_entry=wiki_e, overwrite_template_file=True, epochs=0, num_of_templates=100000, gw_prob=0.5, right_ascension=[-np.pi, np.pi], declination=[-np.pi/2, np.pi/2], polarization=[0.0, np.pi], resample_delta_t=(1.0/2048, 1.0/512, 1.0/128), resample_t_len=(2.0, 8.0, 32.0), custom_message='Generated small template file at more realistic SNRs with varied sky position.')
    #except:
        #pass
    
    #try:
        #run_net('collect_inception_net_6_rev_6', 'small_snr_vary_sky_masses_3_100000', ini_file='testing_net.ini', snr=[8.0, 15.0], create_wiki_entry=wiki_e, overwrite_template_file=True, epochs=0, num_of_templates=100000, gw_prob=0.5, right_ascension=[-np.pi, np.pi], declination=[-np.pi/2, np.pi/2], polarization=[0.0, np.pi], mass1=[1.3, 1.5], mass2=[1.3, 1.5], resample_delta_t=(1.0/2048, 1.0/512, 1.0/128), resample_t_len=(2.0, 8.0, 32.0),  custom_message='Generated small template file at more realistic SNRs with varied sky position and varied masses.')
    #except:
        #pass
    
    #run_net('inception_net_2', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=True, overwrite_template_file=False, epochs=ep, num_of_templates=20000, format_data=True, epoch_break=5, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32)
    
    #run_net('inception_net_4', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=True, overwrite_template_file=False, epochs=ep, num_of_templates=20000, format_data=True, epoch_break=5, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32)
    
    #run_net('inception_net_6', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=True, overwrite_template_file=False, epochs=ep, num_of_templates=20000, format_data=True, epoch_break=5, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32)
    
    #run_net('inception_net_8', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=True, overwrite_template_file=False, epochs=ep, num_of_templates=20000, format_data=True, epoch_break=5, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32)
    
    #try:
        #run_net('inception_net_10', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=wiki_e, overwrite_template_file=False, epochs=1000, num_of_templates=20000, format_data=True, epoch_break=5, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32)
    #except:
        #pass
    
    #try:
        #run_net('inception_net_6_skip', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=wiki_e, overwrite_template_file=False, epochs=300, num_of_templates=20000, format_data=True, epoch_break=5, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32, custom_message='This net takes as input the channels 2048Hz, 512Hz, 128Hz. Reduced the depth of the model to 4 inception layers to see if that improves things. The reason for this decision is the results from 05.04.2019 as they can be found in the wiki.')
    #except:
        #pass
    
    #run_net('collect_inception_net_6_rev_6', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=wiki_e, overwrite_template_file=False, epochs=ep, num_of_templates=200, format_data=True, epoch_break=3, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32, custom_message='This network has the same architecture  collect_inception_net_6_rev_4, but introduces dilation to the convolution layers. Specifically it dilates the three inception layers in each tower. They now each have a kernel size of 4, but have dilation rates of 1, 2, 3 in each parallel convolution.')
    
    #run_net('collect_inception_net_6_rev_3', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=wiki_e, overwrite_template_file=False, epochs=ep, num_of_templates=200, format_data=True, epoch_break=3, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32, custom_message='This network in architecture is close to the inception_net_6_rev_2, but removes two inception layers in the lowest frequency channel, such that all stacks are of the same height.')
    
    #run_net('collect_inception_net_6_rev_4', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=wiki_e, overwrite_template_file=False, epochs=ep, num_of_templates=200, format_data=True, epoch_break=3, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32, custom_message='This network in architecture is close to the inception_net_6_rev_3, but concatenates the stacks after only three inception layers on each stack. Afterwards two more inception layers were added to help learning.')
    
    #try:
        #run_net('inception_net_6_skip', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=wiki_e, overwrite_template_file=False, epochs=300, num_of_templates=20000, format_data=True, epoch_break=5, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32, custom_message='This net takes as input the channels 2048Hz, 512Hz, 128Hz. Reduced the depth of the model to 4 inception layers to see if that improves things. The reason for this decision is the results from 05.04.2019 as they can be found in the wiki.')
    #except:
        #pass
    
    #try:
        #run_net('inception_net_2_small', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=wiki_e, overwrite_template_file=False, epochs=ep, num_of_templates=20000, format_data=True, epoch_break=5, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32)
    #except:
        #pass
    
    #try:
        #run_net('inception_net_4_small', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=wiki_e, overwrite_template_file=False, epochs=ep, num_of_templates=20000, format_data=True, epoch_break=5, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32)
    #except:
        #pass
    
    #try:
        #run_net('inception_net_6_small', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=wiki_e, overwrite_template_file=False, epochs=ep, num_of_templates=20000, format_data=True, epoch_break=5, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32)
    #except:
        #pass
    
    #try:
        #run_net('inception_net_8_small', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=wiki_e, overwrite_template_file=False, epochs=ep, num_of_templates=20000, format_data=True, epoch_break=5, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32)
    #except:
        #pass
    
    #try:
        #run_net('inception_net_10_small', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=wiki_e, overwrite_template_file=False, epochs=ep, num_of_templates=20000, format_data=True, epoch_break=5, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32)
    #except:
        #pass
    
    #try:
        #run_net('inception_net_12_small', 'mult_output_data_medium_small', ini_file='testing_net.ini', snr=[10.0, 50.0], create_wiki_entry=wiki_e, overwrite_template_file=False, epochs=ep, num_of_templates=20000, format_data=True, epoch_break=5, gw_prob=0.5, use_custom_compilation=True, show_snr_plot=False, batch_size=32)
    #except:
        #pass
    
