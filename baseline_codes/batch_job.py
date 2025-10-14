import subprocess
from subprocess import PIPE


for ds in ['School','Chemical','Landmine','Parkinsons']:
    for ClusterType in ['KMeans']:
        Type = 'WeightMatrix'
        #if ds == 'School':
        #    continue
        if ClusterType == 'Hierarchical':
            for range_x in ['first','sec']:
                script_name = f"clustering_MTLs_{ds.lower()}.py"
                run_params = ['sbatch', 'srun_baseline.sh'] + [script_name, ClusterType, Type, '1', range_x]
                print(f"Running...{ds} {ClusterType} {Type} {range_x}")
                completed = subprocess.run(run_params, stdout=PIPE)
        else:
            script_name = f"clustering_MTLs_{ds.lower()}.py"
            run_params = ['sbatch', 'srun_baseline.sh'] + [script_name, ClusterType, Type, '2']
            print(f"Running...{ds} {ClusterType} {Type}")
            completed = subprocess.run(run_params, stdout=PIPE)
