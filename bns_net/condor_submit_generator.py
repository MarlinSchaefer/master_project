import numpy as np
import subprocess
import os
from progress_bar import progress_tracker

def get_submit_file(highLevel, lowLevel):
    lines=[
    '## Condor submit file for evaluating network using final_evaluation.py',
    'Universe        = vanilla',
    'Executable      = /work/marlin.schaefer/master_project/bns_net/final_evaluation.py',
    'Arguments       = {} {}'.format(highLevel, lowLevel),
    'Environment     = NRUNS=1;JOB_LABEL=_Cl$(Cluster)_Job$(Process);',
    'getenv          = True',
    'should_transfer_files = YES',
    'when_to_transfer_output = ON_EXIT_OR_EVICT',
    'transfer_executable = False',
    'transfer_output_files = ./',
    '',
    'output          = /work/marlin.schaefer/job_outputs/out.Cl$(Cluster).$(Process).{}.{}'.format(highLevel, lowLevel),
    'error           = /work/marlin.schaefer/job_outputs/err.Cl$(Cluster).$(Process).{}.{}'.format(highLevel, lowLevel),
    'log             = /work/marlin.schaefer/job_outputs/log.Cl$(Cluster).$(Process).{}.{}'.format(highLevel, lowLevel),
    '',
    '',
    'request_memory = 10000',
    '',
    'on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)',
    '',
    'accounting_group = aei.dev.ml',
    'accounting_group_user = marlin.schaefer',
    '',
    '## use fastest available machines',
    'Rank            = kflops',
    '',
    '## send email?',
    'notification    = Never',
    '##notify_user   = nix@aei.mpg.de',
    '',
    'Queue   1'
    ]
    return lines

def write_submit_file(highLevel, lowLevel):
    file_path = os.path.join('/work/marlin.schaefer', 'condor_submit_final_evaluation_{}_{}.sub'.format(highLevel, lowLevel))
    with open(file_path, 'w') as f:
        for line in get_submit_file(highLevel, lowLevel):
            f.write(line + '\n')
    return file_path

def main():
    highLevelSteps = np.arange(3, 202, 2, dtype=int)
    lowLevelSteps = np.arange(22, dtype=int)
    bar = progress_tracker(len(highLevelSteps) * len(lowLevelSteps), name='Submitting jobs')
    for highLevel in highLevelSteps:
        for lowLevel in lowLevelSteps:
            condor_file = write_submit_file(highLevel, lowLevel)
            subprocess.call(["condor_submit", condor_file])
            bar.iterate()
    return

if __name__ == "__main__":
    main()
