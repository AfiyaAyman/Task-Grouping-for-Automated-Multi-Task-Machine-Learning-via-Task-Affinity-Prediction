#!/bin/bash

#SBATCH -J DT_TGP
#SBATCH -t 200:00:00
#SBATCH -N 1 -n 16
#SBATCH --mem=64g
#SBATCH -A joshi


#module load python/3.8
eval "$(conda shell.bash hook)"
conda activate AutoML
python cluster_training_DT.py "$@"
