#!/bin/bash

#SBATCH --job-name=cuda-practice
#SBATCH --account=xlzd

#SBATCH --partition=gpu
#SBATCH --mem=100M
#SBATCH --time=0:05:00           # Run time

#SBATCH --nodes=1                # Number of CPUs
#SBATCH --ntasks=1               # Number of processes to run
#SBATCH --cpus-per-task=1        # Number of threads per task
#SBATCH --gres=gpu:1             # Number of GPUs per node

#SBATCH --output=/veracruz/projects/x/xlzd/CUDA-examples/slurm/slurm.out
#SBATCH --error=/veracruz/projects/x/xlzd/CUDA-examples/slurm/slurm.err

executable=$1
echo "Submitting job using: $executable"

source /veracruz/projects/x/xlzd/CUDA-examples/slurm/setup.sh
./"$executable"

echo "Job completed."
