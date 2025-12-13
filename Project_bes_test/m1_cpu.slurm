#!/bin/bash
#SBATCH --job-name="Milestone1_CPU"
#SBATCH --output="Milestone1_CPU.out"
#SBATCH --error="Milestone1_CPU.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:25:00

module reset
module load cuda/12.4

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

# srun ./m1_cpu 100 > m1_cpu.out
# srun ./m1_cpu 1000 && gprof -Q ./m1_cpu gmon.out > cpu_gprof.txt

# Generate running time information
# { time srun ./m1_cpu 100 > m1_cpu_100.out; } 2> m1_cpu_time.out


{ time srun ./m1_cpu 100 > m1_cpu_100.out; } 2> m1_cpu_time_100.out
{ time srun ./m1_cpu 1000 > m1_cpu_1000.out; } 2> m1_cpu_time_1000.out
{ time srun ./m1_cpu 10000 > m1_cpu_10000.out; } 2> m1_cpu_time_10000.out
