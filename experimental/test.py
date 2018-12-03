from run_net import run_net

if __name__ == "__main__":
    run_net('test_net', 'test', num_of_templates=20, mass1=[10.0,45.0], overwrite_template_file=False, ignore_fixable_errors=True)
