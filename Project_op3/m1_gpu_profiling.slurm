#!/bin/bash
#SBATCH --job-name="Milestone1_CPU"
#SBATCH --output="Milestone1_CPU.out"
#SBATCH --error="Milestone1_CPU.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects,perf,nvperf"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:25:00

module reset
module load cuda

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

# Basic GPU NSys Profiling
srun nsys profile --stats=true ./m1_gpu > m1_gpu_profile.out

srun ncu --set full -f -o analysis_file_m1_gpu ./m1_gpu 100 > m1_gpu_ncu.out