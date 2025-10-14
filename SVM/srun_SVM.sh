#!/bin/bash

#SBATCH -J LM_Exp
#SBATCH -t 200:00:00
#SBATCH -N 1 -n 20
#SBATCH --mem=128g
#SBATCH -A joshi


#module load python/3.8
eval "$(conda shell.bash hook)"
conda activate AutoML
python cluster_training_SVM.py "$@"
