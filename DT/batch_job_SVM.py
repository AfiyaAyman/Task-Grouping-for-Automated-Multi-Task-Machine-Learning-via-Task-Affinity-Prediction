import subprocess
from subprocess import PIPE
import ast

import pandas as pd
for ds in ['School','Chemical','Landmine']:#,'Parkinsons']:
    for v in range(0,1,50):
        v = str(v)
    run_params = ['sbatch', 'srun_SVM.sh'] + [ds, '1']
    print(f"Running...{ds}")
    completed = subprocess.run(run_params, stdout=PIPE)
