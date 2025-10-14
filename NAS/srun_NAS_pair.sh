#!/bin/bash

#SBATCH -J SVM_NAS
#SBATCH -t 100:00:00
#SBATCH -N 1 -n 20
#SBATCH --mem=128g
#SBATCH -A joshi


#module load python/3.8
eval "$(conda shell.bash hook)"
conda activate AutoML
python NAS_TGP_ALL.py "$@"
#python NAS_pairwise.py "$@"
