#!/bin/bash
#SBATCH --job-name="lab6"
#SBATCH --output="lab6.out"
#SBATCH --error="lab6.err"
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:01:00

module reset
module load cuda/12.4

set -e

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

SANITIZER_FLAGS="--show-backtrace no --print-limit 1 --error-exitcode -1 --destroy-on-device-error kernel"

srun echo -e "\nmemcheck tests:"
srun compute-sanitizer --tool memcheck $SANITIZER_FLAGS lab6 -e data/0/output.raw -i data/0/input0.raw -o /tmp/myoutput.raw -t vector
srun compute-sanitizer --tool memcheck $SANITIZER_FLAGS lab6 -e data/1/output.raw -i data/1/input0.raw -o /tmp/myoutput.raw -t vector
srun compute-sanitizer --tool memcheck $SANITIZER_FLAGS lab6 -e data/2/output.raw -i data/2/input0.raw -o /tmp/myoutput.raw -t vector
srun compute-sanitizer --tool memcheck $SANITIZER_FLAGS lab6 -e data/3/output.raw -i data/3/input0.raw -o /tmp/myoutput.raw -t vector
srun compute-sanitizer --tool memcheck $SANITIZER_FLAGS lab6 -e data/4/output.raw -i data/4/input0.raw -o /tmp/myoutput.raw -t vector
srun compute-sanitizer --tool memcheck $SANITIZER_FLAGS lab6 -e data/5/output.raw -i data/5/input0.raw -o /tmp/myoutput.raw -t vector
srun compute-sanitizer --tool memcheck $SANITIZER_FLAGS lab6 -e data/6/output.raw -i data/6/input0.raw -o /tmp/myoutput.raw -t vector
srun compute-sanitizer --tool memcheck $SANITIZER_FLAGS lab6 -e data/7/output.raw -i data/7/input0.raw -o /tmp/myoutput.raw -t vector
srun compute-sanitizer --tool memcheck $SANITIZER_FLAGS lab6 -e data/8/output.raw -i data/8/input0.raw -o /tmp/myoutput.raw -t vector
srun compute-sanitizer --tool memcheck $SANITIZER_FLAGS lab6 -e data/9/output.raw -i data/9/input0.raw -o /tmp/myoutput.raw -t vector

srun echo -e "\ninitcheck tests:"
srun compute-sanitizer --tool initcheck $SANITIZER_FLAGS lab6 -e data/0/output.raw -i data/0/input0.raw -o /tmp/myoutput.raw -t vector
srun compute-sanitizer --tool initcheck $SANITIZER_FLAGS lab6 -e data/1/output.raw -i data/1/input0.raw -o /tmp/myoutput.raw -t vector
srun compute-sanitizer --tool initcheck $SANITIZER_FLAGS lab6 -e data/2/output.raw -i data/2/input0.raw -o /tmp/myoutput.raw -t vector
srun compute-sanitizer --tool initcheck $SANITIZER_FLAGS lab6 -e data/3/output.raw -i data/3/input0.raw -o /tmp/myoutput.raw -t vector
srun compute-sanitizer --tool initcheck $SANITIZER_FLAGS lab6 -e data/4/output.raw -i data/4/input0.raw -o /tmp/myoutput.raw -t vector
srun compute-sanitizer --tool initcheck $SANITIZER_FLAGS lab6 -e data/5/output.raw -i data/5/input0.raw -o /tmp/myoutput.raw -t vector
srun compute-sanitizer --tool initcheck $SANITIZER_FLAGS lab6 -e data/6/output.raw -i data/6/input0.raw -o /tmp/myoutput.raw -t vector
srun compute-sanitizer --tool initcheck $SANITIZER_FLAGS lab6 -e data/7/output.raw -i data/7/input0.raw -o /tmp/myoutput.raw -t vector
srun compute-sanitizer --tool initcheck $SANITIZER_FLAGS lab6 -e data/8/output.raw -i data/8/input0.raw -o /tmp/myoutput.raw -t vector
srun compute-sanitizer --tool initcheck $SANITIZER_FLAGS lab6 -e data/9/output.raw -i data/9/input0.raw -o /tmp/myoutput.raw -t vector

srun echo -e "\nsynccheck tests:"
srun compute-sanitizer --tool synccheck $SANITIZER_FLAGS lab6 -e data/0/output.raw -i data/0/input0.raw -o /tmp/myoutput.raw -t vector
srun compute-sanitizer --tool synccheck $SANITIZER_FLAGS lab6 -e data/1/output.raw -i data/1/input0.raw -o /tmp/myoutput.raw -t vector
srun compute-sanitizer --tool synccheck $SANITIZER_FLAGS lab6 -e data/2/output.raw -i data/2/input0.raw -o /tmp/myoutput.raw -t vector
srun compute-sanitizer --tool synccheck $SANITIZER_FLAGS lab6 -e data/3/output.raw -i data/3/input0.raw -o /tmp/myoutput.raw -t vector
srun compute-sanitizer --tool synccheck $SANITIZER_FLAGS lab6 -e data/4/output.raw -i data/4/input0.raw -o /tmp/myoutput.raw -t vector
srun compute-sanitizer --tool synccheck $SANITIZER_FLAGS lab6 -e data/5/output.raw -i data/5/input0.raw -o /tmp/myoutput.raw -t vector
srun compute-sanitizer --tool synccheck $SANITIZER_FLAGS lab6 -e data/6/output.raw -i data/6/input0.raw -o /tmp/myoutput.raw -t vector
srun compute-sanitizer --tool synccheck $SANITIZER_FLAGS lab6 -e data/7/output.raw -i data/7/input0.raw -o /tmp/myoutput.raw -t vector
srun compute-sanitizer --tool synccheck $SANITIZER_FLAGS lab6 -e data/8/output.raw -i data/8/input0.raw -o /tmp/myoutput.raw -t vector
srun compute-sanitizer --tool synccheck $SANITIZER_FLAGS lab6 -e data/9/output.raw -i data/9/input0.raw -o /tmp/myoutput.raw -t vector

sync

srun echo -e "\nall tests completed."
