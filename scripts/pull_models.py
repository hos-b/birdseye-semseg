import os
import sys
from data.config import EvaluationConfig
from pathlib import Path

if __name__ == '__main__':
    try:
        remote_address = os.environ['REMOTE_MCNN_MODEL_PATH']
    except KeyError:
        print('could not find REMOTE_MCNN_MODEL_PATH enviornment variable')
        exit(-1)
    
    # pull only the models used for evaluation
    if len(sys.argv) == 1:
        run_path = os.path.join(os.getcwd(), 'runs')
        if not os.path.exists(run_path):
            os.makedirs(run_path)
        print(f'pulling evaluation models')
        eval_cfg = EvaluationConfig('config/evaluation.yml')
        for eval_run in eval_cfg.runs:
            print(f'-----------------------------------')
            print(f'pulling {eval_run}')
            ret_val = os.system(f'scp -r "{remote_address}/{eval_run}" ./runs > /dev/null 2>&1')
            if ret_val != 0:
                print(f'failed to pull {eval_run}')
                exit()
        print(f'-----------------------------------')
    # pull all models
    elif len(sys.argv) == 2 and (sys.argv[1] == '-a' or sys.argv[1] == '--all'):
        print(f'pulling all models')
        os.system(f'scp -r {remote_address} ./runs > /dev/null 2>&1')

    print('done')