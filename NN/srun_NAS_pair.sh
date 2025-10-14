#!/bin/bash

#SBATCH -J NN_LM
#SBATCH -t 100:00:00
#SBATCH -N 1 -n 20
#SBATCH --mem=64g
#SBATCH -A joshi


#module load python/3.8
eval "$(conda shell.bash hook)"
conda activate AutoML
python landmine_QR.py "$@"
#python NAS_pairwise.py "$@"
