#! /bin/bash
#SBATCH -J G.O.R.A
#SBATCH --mem=32768
#SBATCH --nodelist=cn4
#SBATCH -p ai_gpu
#SBATCH --time 90:00:00
#SBATCH --gres=gpu:1

source /dev/shm/okirnap/setup.sh
julia train.jl > stack_exp3.txt
