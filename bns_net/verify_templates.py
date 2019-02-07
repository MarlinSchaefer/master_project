from scipy.stats import normaltest
from ini_handeling import run_net_defaults, make_template_bank_defaults
from run_net import filter_keys
from load_data import load_parameter_space, load_training_labels, load_training_calculated_snr, load_testing_labels, load_testing_calculated_snr

def verify_templates(file_path):
    tr_labels = load_training_labels()
    tr_snr = load_training_calculated_snr()
    te_labels = load_testing_labels()
    te_snr = load_testing_calculated_snr()
    
    total_labels = list(tr_labels) + list(te_labels)
    del tr_labels
    del te_labels
    total_snr = list(tr_snr) + list(te_snr)
    del tr_snr
    del te_snr
    
    THRESHOLD = 0.4
    
    diff = [total_labels[i] - total_snr[i] for i in range(len(total_labels))]
    del total_labels
    del total_snr
    
    stats, pval = normaltest(diff)
    
    return(pval > THRESHOLD)

def check_input(file_path, **kwargs):
    used_data = dict(kwargs)
    run_net_def = run_net_defaults()
    wav_arg, opt_arg = make_template_bank_defaults()
        
    run_net_def, used_data = filter_keys(run_net_def, used_data)
    wav_arg, used_data = filter_keys(wav_arg, used_data)
    opt_arg, used_data = filter_keys(opt_arg, used_data)
    
    used_data.update(run_net_def)
    del run_net_def
    used_data.update(wav_arg)
    del wav_arg
    used_data.update(opt_arg)
    del opt_arg
    
    file_data = load_parameter_space(file_path)
    
    for key, val in file_data.items():
        if key in used_data:
            #Protect against mismatched array length
            try:
                if not (val == used_data[key]).all():
                    return(False)
            except AttributeError:
                return(False)
        else:
            return(False)
    
    return(True)

