import subprocess
from subprocess import PIPE

for ds in ['School']:#,'Chemical','Landmine','Parkinsons']:
    for i in range(1,6):
        run_params = ['sbatch', 'srun_NAS_pair.sh'] + [str(i)]
        print(f"Running...{ds}")
        completed = subprocess.run(run_params, stdout=PIPE)
