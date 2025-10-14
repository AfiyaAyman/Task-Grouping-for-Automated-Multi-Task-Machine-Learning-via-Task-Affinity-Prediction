import subprocess
from subprocess import PIPE
import ast

import pandas as pd
for ds in ['Landmine']:
    for TYPE in ['Exponential']:
    #    v = str(v)
        for v in range(10,100,10):
            v_range = str(v)
            run_params = ['sbatch', 'srun_SVM.sh'] + [ds, '1', 'Hierarchical', TYPE, v_range]
            print(f"Running...{ds}")
            completed = subprocess.run(run_params, stdout=PIPE)
