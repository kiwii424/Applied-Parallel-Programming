#!/bin/bash
#SBATCH --job-name="Milestone2"
#SBATCH --output="Milestone2.out"
#SBATCH --error="Milestone2.err"
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

outdir="./test1"

module reset
module load cuda/12.4

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

timestamp=$(date "+%Y-%m-%d %H:%M:%S")
jobid=$SLURM_JOB_ID


run_batch() {
    batch=$1
    outfile="${outdir}/m2_fused_${batch}.out"

    : > "$outfile"

    timestamp=$(date "+%Y-%m-%d %H:%M:%S")

    {
        echo "====================================="
        echo " Time : $timestamp"
        echo " JobID: $SLURM_JOB_ID"
        echo "====================================="
    } >> $outfile

    { time srun ./m2_fused $batch >> $outfile; } 2>> $outfile
}

run_batch 100
run_batch 1000
run_batch 10000

# Sanitizer (separate file)
srun compute-sanitizer ./m2_fused 100 > ${outdir}/m2_fused_sanitizer.out
