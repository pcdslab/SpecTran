#!/bin/bash

#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=test_mar23
#SBATCH --error=/disk/dragon-storage/homes/seber007/test_mar23.%J.err
#SBATCH --output=/disk/dragon-storage/homes/seber007/test_mar23.%J.out

module load cuda-11.6.0-gcc-9.4.0-system

srun -N 1 -G 2 --ntasks-per-node=1 --cpu-bind=sockets python /disk/dragon-storage/homes/seber007/fiu_repos/pcdslab/SpecTran/train_specbert_test_slurm.py
