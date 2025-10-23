#!/bin/bash
#SBATCH --job-name="ECE 408 Profiling Example"
#SBATCH --output="ncu.out"
#SBATCH --error="ncu.err"
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="perf,nvperf"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:02:00

module reset
module load cuda/12.4

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

srun bash generate_ncu-rep.sh
