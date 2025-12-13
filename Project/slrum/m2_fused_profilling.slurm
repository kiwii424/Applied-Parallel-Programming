#!/bin/bash
#SBATCH --job-name="Milestone3"
#SBATCH --output="Milestone3.out"
#SBATCH --error="Milestone3.err"
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

outdir="./Milestone2_output/profile_output"

module reset
module load cuda/12.4

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"


# Generate nsys profile information, output will be in profile.out and report.nsys-rep
# srun nsys profile --stats=true ./m3 > ${outdir}/m3_nsys_profile.out
srun nsys profile --stats=true -o ${outdir}/nsys_report ./m2_fused 10000 > ${outdir}/m2_fused_nsys_profile.out

# Generate ncu profile information, output will be in analysis_file.ncu-rep
srun ncu --set full -f -o ${outdir}/analysis_file ./m2_fused 10000 > ${outdir}/m2_fused_ncu.out