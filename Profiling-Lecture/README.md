# ECE-408-Profiling-Lecture
Contains kernel source code, slurm script for profiling, and generated ncu and nsys files

## build the kernels
To build the project, run:
```bash
bash build.sh
```

## run the kernels
To run the project, run:
```bash
sbatch run.slurm
```

## use nsys to profile
To run nsys, run:
```bash
sbatch nsys.slurm
```

## use ncu to profile
To run ncu, run
```
sbatch ncu.slurm
```
