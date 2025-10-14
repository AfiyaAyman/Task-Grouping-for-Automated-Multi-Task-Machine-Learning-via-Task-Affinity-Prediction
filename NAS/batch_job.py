import subprocess
from subprocess import PIPE

for ds in ['School','Chemical','Landmine','Parkinsons']:
    run_params = ['sbatch', 'srun_NAS_pair.sh'] + [ds, 'SVM']
    print(f"Running...{ds}")
    completed = subprocess.run(run_params, stdout=PIPE)
