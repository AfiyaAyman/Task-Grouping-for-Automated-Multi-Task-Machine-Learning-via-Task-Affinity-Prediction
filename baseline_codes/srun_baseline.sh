#!/bin/bash

#SBATCH -J KMM_Cluster
#SBATCH -t 200:00:00
#SBATCH -N 1 -n 32
#SBATCH --mem=128g
#SBATCH -A joshi


#module load python/3.8
eval "$(conda shell.bash hook)"
conda activate AutoML
python "$@"
