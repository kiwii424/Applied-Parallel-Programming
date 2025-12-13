#!/bin/bash
#SBATCH --job-name="Milestone3"
#SBATCH --output="Milestone3.out"
#SBATCH --error="Milestone3.err"
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

outdir="./project/m3/req_0"

module reset
module load cuda/12.4


echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"


jobid=$SLURM_JOB_ID


run_batch() {
    batch=$1
    outfile="${outdir}/m3_${batch}.out"

    : > "$outfile"

    timestamp=$(date "+%Y-%m-%d %H:%M:%S")

    {
        echo "====================================="
        echo " Time : $timestamp"
        echo " JobID: $SLURM_JOB_ID"
        echo "====================================="
    } >> $outfile

    { time srun ./m3 $batch >> $outfile; } 2>> $outfile
}

run_batch 100
run_batch 1000
run_batch 10000

# Sanitizer (separate file)
srun compute-sanitizer ./m3 100 > ${outdir}/m3_sanitizer.out