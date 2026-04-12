# Applied-Parallel-Programming

ECE 408 / CS 483 / CSE 408 (fa25) by UIUC, IL, USA
Instructor: Volodymyr Kindratenko

CUDA / GPU programming portfolio from UIUC's Applied Parallel Programming course, featuring profiling-driven implementations of convolution, dense and sparse linear algebra, reduction, scan, and CNN inference.

The fastest path to review this repo is:

1. [Final Project](#final-project)
2. [Lab 3](#lab-3), [Lab 4](#lab-4), [Lab 6](#lab-6), [Lab 7](#lab-7), and [Lab 8](#lab-8)

## What This Repo Demonstrates

- CUDA host/device programming: memory allocation, transfers, launch configuration, synchronization, and cleanup
- GPU memory hierarchy usage: shared memory, constant memory, pinned memory, and register tiling
- Parallel algorithm design: vector ops, GEMM, convolution, histogramming, reduction, scan, and sparse matrix-vector multiplication
- Performance engineering workflow: layout transforms (`im2col`), kernel fusion, streams, cuBLAS, FP16, and WMMA / Tensor Core experiments
- HPC tooling: Slurm-based execution on Delta, plus profiling workflows with `gprof`, `nsys`, and `ncu`

## Summary


| Lab                             | Problem                                                                        | What I built                                                                                                                                                                               | Outcome                                                                                                                   |
| ------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| [Final Project](#final-project) | Accelerate convolution layers in a modified LeNet-style CNN inference pipeline | CPU baseline, direct CUDA convolution,`im2col` unrolling, fused convolution, streams, Tensor Core experiments, cuBLAS / FP16 / memory-hierarchy optimizations, and profiling-driven tuning | Final performance time`38.49 ms` vs baseline `60 ms`, while documenting tradeoffs across multiple optimization strategies |
| [Lab 0](#lab-0)                 | Set up CUDA development on Delta and inspect GPU capabilities                  | Device-query workflow and Slurm-based execution setup                                                                                                                                      | Established an HPC-first workflow and hardware-awareness baseline                                                         |
| [Lab 1](#lab-1)                 | Vector addition                                                                | My first complete CUDA program with explicit host/device memory management and a custom kernel                                                                                             | Passed all 10 provided test cases                                                                                         |
| [Lab 2](#lab-2)                 | Basic dense matrix multiplication                                              | A baseline GEMM kernel with correct indexing and boundary handling for non-square inputs                                                                                                   | Passed all 10 provided test cases                                                                                         |
| [Lab 3](#lab-3)                 | Tiled dense matrix multiplication                                              | A shared-memory tiled GEMM kernel to improve data reuse and reduce global memory traffic                                                                                                   | Passed all 10 provided test cases                                                                                         |
| [Lab 4](#lab-4)                 | 3D convolution                                                                 | A 3D convolution kernel using constant memory for filter weights and boundary-safe indexing                                                                                                | Passed all 6 provided volumetric test cases                                                                               |
| [Lab 5](#lab-5)                 | Histogram equalization                                                         | A multi-stage image-processing pipeline across GPU kernels plus a CPU-side CDF scan                                                                                                        | Passed all 10 provided image test cases                                                                                   |
| [Lab 6](#lab-6)                 | Parallel reduction                                                             | A shared-memory reduction kernel with host-side final accumulation                                                                                                                         | Passed correctness plus sanitizer checks with 0 reported errors                                                           |
| [Lab 7](#lab-7)                 | Parallel scan                                                                  | A hierarchical multi-kernel prefix-sum implementation with block-sum propagation                                                                                                           | Passed correctness plus sanitizer checks with 0 reported errors                                                           |
| [Lab 8](#lab-8)                 | Sparse matrix-vector multiplication (JDS)                                      | A CUDA SpMV kernel over the JDS sparse format with row permutation handling                                                                                                                | Passed all 7 provided sparse test cases                                                                                   |

<a id="final-project"></a>

## Final Project

- Goal: optimize the convolution layers in a modified LeNet-5 inference pipeline on Fashion-MNIST with CUDA.
- Built: CPU reference, direct CUDA convolution, `im2col` unrolling, fused convolution, and milestone-3 optimization branches.
- Explored optimizations: streams, constant memory, `__restrict__`, loop unrolling, launch-parameter sweeps, cuBLAS, Tensor Core / WMMA, FP16, and register/shared-memory tiling.
- Profiling workflow: used `gprof`, `nsys`, and `ncu` to compare implementations, inspect overlap, and identify memory, occupancy, and kernel bottlenecks.
- Engineering takeaway: I did not stop at one optimization. I tested multiple paths, measured tradeoffs, and kept the combination that produced the best end-to-end result.
- Final result: best runtime `38.49 ms` vs baseline `60 ms`, a `35.9%` reduction.
- Key files: [cpu-new-forward.cc](Project/project/src/layer/custom/cpu-new-forward.cc), [new-forward.cu](Project/project/src/layer/custom/new-forward.cu), [unroll-new-forward.cu](Project/project/src/layer/custom/unroll-new-forward.cu), [kernel-fusion-forward.cu](Project/project/src/layer/custom/kernel-fusion-forward.cu), [m3-forward.cu](Project/project/src/layer/custom/m3-forward.cu), [milestone-3 experiment variants](Project/project/m3)

Final report: [ECE408_FA25_mh126_final_report.pdf](Project/ECE408_FA25_mh126_final_report.pdf)

## Project Structure

- [Project](Project): final project code, milestone variants, Slurm scripts, and report.
- `lab0` to `lab8`: course labs covering core CUDA programming, memory hierarchy, parallel primitives, and sparse computation.
- [Profiling-Lecture](Profiling-Lecture): supplemental profiling examples and reference material for `nsys` / `ncu`.
- Main implementation work is concentrated in the CUDA source files inside each lab folder and `Project/project/src/layer/custom`.

## Environment / How to Run

- Primary environment: NCSA Delta cluster, `CUDA 12.4`, Slurm, and A40-class GPU partitions used by the course scripts.
- Example lab workflow: `cd lab3 && make && sbatch job.slurm`
- Example final-project workflow: `cd Project && ./run.sh build && sbatch slrum/m3.slurm`
- Profiling workflow: `gprof` for CPU, `nsys` for timeline/system analysis, and `ncu` for kernel-level analysis.

<a id="lab-0"></a>

## Lab 0

- Problem: set up CUDA development and inspect the target GPU environment on Delta.
- Built: a device-query workflow for GPU capability, memory, and execution limits.
- Outcome: established the HPC workflow used by every later lab and project.
- Implementation: [lab0.cu](lab0/lab0.cu)

<a id="lab-1"></a>

## Lab 1

- Problem: vector addition.
- Built: my first end-to-end CUDA program with explicit memory management, launch configuration, synchronization, and a custom kernel.
- Outcome: passed all 10 provided cases.
- Implementation: [lab1.cu](lab1/lab1.cu)
- Validation: [lab1.out](lab1/lab1.out)

<a id="lab-2"></a>

## Lab 2

- Problem: basic dense matrix multiplication.
- Built: a baseline GEMM kernel with correct indexing and boundary handling for non-square inputs.
- Outcome: passed all 10 provided cases.
- Implementation: [lab2.cu](lab2/lab2.cu)
- Validation: [lab2.out](lab2/lab2.out)

<a id="lab-3"></a>

## Lab 3

- Problem: tiled dense matrix multiplication.
- Built: a shared-memory GEMM kernel to improve tile reuse and reduce global-memory traffic.
- Outcome: passed all 10 provided cases.
- Implementation: [lab3.cu](lab3/lab3.cu)
- Validation: [lab3.out](lab3/lab3.out)

<a id="lab-4"></a>

## Lab 4

- Problem: 3D convolution.
- Built: a 3D convolution kernel with constant-memory filter storage and boundary-safe indexing.
- Outcome: passed all 6 provided volumetric cases.
- Implementation: [lab4.cu](lab4/lab4.cu)
- Validation: [lab4.out](lab4/lab4.out)

<a id="lab-5"></a>

## Lab 5

- Problem: histogram equalization.
- Built: a multi-stage image-processing pipeline across GPU kernels, with the CDF scan handled on CPU.
- Outcome: passed all 10 provided image cases.
- Implementation: [lab5.cu](lab5/lab5.cu)
- Validation: [lab5.out](lab5/lab5.out)

<a id="lab-6"></a>

## Lab 6

- Problem: parallel reduction.
- Built: a shared-memory block reduction kernel plus host-side final accumulation.
- Outcome: passed correctness tests and `compute-sanitizer` checks with `0` reported errors.
- Implementation: [lab6.cu](lab6/lab6.cu)
- Validation: [lab6.out](lab6/lab6.out)

<a id="lab-7"></a>

## Lab 7

- Problem: parallel scan.
- Built: a hierarchical prefix-sum pipeline with per-block scan, block-sum scan, and final add-back.
- Outcome: passed correctness tests and `compute-sanitizer` checks with `0` reported errors.
- Implementation: [lab7.cu](lab7/lab7.cu)
- Validation: [lab7.out](lab7/lab7.out)

<a id="lab-8"></a>

## Lab 8

- Problem: sparse matrix-vector multiplication in JDS format.
- Built: a CUDA SpMV kernel with row permutation and jagged-diagonal traversal.
- Outcome: passed all 7 provided sparse cases.
- Implementation: [lab8.cu](lab8/lab8.cu)
- Validation: [lab8.out](lab8/lab8.out)

## Acknowledgements

- UIUC `ECE 408 / CS 483 / CSE 408` course staff for the assignments, infrastructure, and profiling guidance.
- Course-provided starter code, datasets, and test harnesses that this portfolio builds on.
- Upstream components and reference code included inside the project subdirectories.

## License

- This repository includes my coursework together with course starter code and third-party components, so licensing may vary by subdirectory.
- See [Project/project/LICENSE](Project/project/LICENSE) and [Profiling-Lecture/LICENSE](Profiling-Lecture/LICENSE) for included license files.
- If you plan to reuse code from this repository, review the relevant subdirectory license first.

## Contributing

- This repository is a personal course portfolio, so external contributions are not expected.
