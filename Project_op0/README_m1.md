# Project Milestone 1: Convolution Implementation - CPU and GPU
***Deadline: October 10th, 2025 8PM***

The 3-day grace period also applies to project milestones. Please check [Canvas](https://canvas.illinois.edu/courses/60979) for our grace period policy.

The table below contains all of the deliverables.

| Deliverables                                                                            |
| --------------------------------------------------------------------------------------- |
| Create a CPU convolution implementation                                                 |
| Create a basic GPU Convolution implementation from Lecture 11                           |
| Correctness and timing with different dataset sizes                                     |
| Submit your work for grading                                                            |


You will edit the following files for milestone 1.
```
project/src/layer/custom/cpu-new-forward.cc
project/src/layer/custom/new-forward.cu
```

**Important Notes: Only modify the files specifically mentioned in this document. Changes to other files will not be used for grading, and may cause unexpected errors that you will be responsible for.**

## Table of Contents

- [Project Setup](#project-setup)
- [Create a CPU Implementation](#create-a-cpu-implementation)
- [Create a GPU Implementation](#create-a-gpu-implementation)
- [Submitting for Grading](#submitting-for-grading)
- [Rubric](#rubric)
- [Appendix](#appendix)

## Project Setup

1. To start, you will need to clone this repository to your folder in the Delta server. Go to your `ece408git` folder and run the following:

   * `git fetch release`
   * `git merge release/main -m "Project checkpoint 1" --allow-unrelated-histories`
   * `git push origin main`

   If you are asked to enter your git credentials (PAT) each time you try to pull or push, you can try to store your git credentials once and for all: just type `git config credential.helper store` in your Delta terminal.

2. We have already set up the dataset for you in the Delta server under the path `/projects/bche/project/data/fmnist-86/`. Please do not modify it!

3. To compile, simply type `./run.sh build`. This will attempt to clean unrelated files and compile your code. If there are any errors, you will see a message in the terminal. Fix the errors and try to recompile again. Please note that the compilation takes place on the cluster head node which has all the tools but does not have any GPUs. Therefore, you cannot execute your compiled code on the head node.

4. Take the CPU implementation as an example, to execute your code, type `sbatch m1_cpu.slurm`. This will schedule the execution of your program on one of the next available compute nodes. The error message during the execution will be input into `Milestone1_CPU.err`. Unlike the head node, compute nodes have GPUs and this is where we want our program to be executed. You will get a message like this `Submitted batch job ID` where the last number is your job ID. Typically, jobs will be executed within a few seconds. However, if your job is not getting executed immediately, you can check its status by typing `squeue --job ID` (do not forget to replace "ID" number with your actual job ID reported by sbatch).

5. To clean, type `./run.sh clean`. This will remove all the files generated during the compilation and execution process.

***Understanding m1_cpu.slurm***

`./m1_cpu 100` runs the code specified in `./project/src/layer/custom/cpu-new-forward.cc` program for a batch of 100 input images.

You should see the following output in m1_cpu.out file:

    Test batch size: 100
    Loading fashion-mnist data...Done
    Loading model...Done
    Conv-CPU==
    Op Time: 1451.97 ms
    Conv-CPU==
    Op Time: 4132.6 ms

    Test Accuracy: 0.86

It is okay for the accuracy to be low here since you haven't implemented the convolutional layers yet.

## Create a CPU Implementation

See the [description](#specifying-the-convolution-implementation) of the skeleton code for a brief overview of what each file does.

Modify `./project/src/layer/custom/cpu-new-forward.cc` to implement the forward convolution described in Chapter 16 of the textbook.
The performance of the CPU convolution is not part of the project evaluation. We only evaluate for correctness.

The algorithm is also below, for your convenience

    for b = 0 .. Batch                     // for each image in the batch
        for m = 0 .. Map_out               // for each output feature maps
            for h = 0 .. Height_out        // for each output element
                for w = 0 .. Width_out
                {
                    output[b][m][h][w] = 0;
                    for c = 0 .. Channel   // sum over all input feature maps
                        for p = 0 .. K // KxK filter
                            for q = 0 .. K
                                output[b][m][h][w] += input[b][c][h + p][w + q] * k[m][c][p][q]
                }

Unlike the convolutions described in the class, note that this one is not centered on the input image. There is no padding and the strides are 1. The following illustration may help you visualize this better.

![ConvExample](https://stanford.edu/~shervine/teaching/cs-230/illustrations/convolution-layer-a.png?1c517e00cb8d709baf32fc3d39ebae67)

*Source: https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks#layer*

Modify `m1_cpu.slurm` to invoke

    srun ./m1_cpu 100 > m1_cpu.out

Please be patient as the CPU implementation is slow and will take several minutes to run. (For instance, a correct implementation with 10k images may take 13+ mins to run). If you want to iterate quickly when developing code using smaller batch sizes, see [Specifying Batch Size](#specifying-batch-size). When your implementation is correct, you should see output like this:

    Test batch size: 100
    Loading fashion-mnist data...Done
    Loading model...Done
    Conv-CPU==
    Op Time: 1451.97 ms
    Conv-CPU==
    Op Time: 4132.6 ms

    Test Accuracy: 0.86


Every time your layer is invoked, it will print the "Op Time," the time spent working on that layer.
Since the network has two convolutional layers, two times will be printed.
You can time the whole program execution by modifying `m1_cpu.slurm` with

    { time srun ./m1_cpu 100 > m1_cpu.out; } 2> time.out

## Create a GPU Implementation

Modify `./project/src/layer/custom/new-forward.cu` to create GPU implementation of the forward convolution. In your template, the host code is separated in 3 parts. `conv_forward_gpu_prolog` allocates memory and copies data from host to device (Note: the device pointers given to you in this function are double pointers). `conv_forward_gpu` computes kernel dimensions and invokes kernel. `conv_forward_gpu_epilog` copies output back to host and free the device memory. You should implement your kernel code from Lecture 11 in `conv_forward_kernel`.

Modify `m1_gpu.slurm` to run with batch_size=100. Run

    srun ./m1_gpu 100 > m1_gpu.out

to run the code specified in `./project/src/layer/custom/new-forward.cu` program for a batch of 100 input images.
If your implementation is correct, it will show the same correctness as the CPU implementation.

The file `m1_gpu.out` includes two performance metrics. "Op time" refers to the time taken by `conv_forward_gpu`. "Layer time" represents the total duration of `conv_forward_gpu_prolog`, `conv_forward_gpu`, and `conv_forward_gpu_epilog` combined.

The sum of Op times on batch_size=10000 should be approximately 70 ms if you implement the basic kernel from Lecture 11 correctly. You must have correct accuracies and total Op time less than 300 ms to earn full credits on the coding part.

`m1_gpu.slurm` will run your code in a single A40 GPU. If you use a different GPU model (such as your personal GPU), the first run may be slower due to JIT caching. For more information, refer to the [Appendix: JIT Caching](#jit-caching).

### Specifying Batch Size

`./m1_cpu` and `./m1_gpu` all take one optional argument: the dataset size.
If the correctness for each possible batch size is as below, you can be reasonably confident your implementation is right. The correctness does depend on the data size.

For example, to check your accuracy on the full data size of 10,000, you could modify the slurm scripts to run

    srun ./m1_cpu 10000 > m1_cpu.out

| Number of Images | Accuracy |
| ---------------- | -------- |
| 100              | 0.86     |
| 1000             | 0.886    |
| 10000            | 0.8714   |

## Submitting for Grading

To submit your work for grading, add, commit, and push your files:

* ```git add -u```
* ```git commit -m "some comment"```
* ```git push origin main```

The code in your GitHub repository at 8pm on Monday, October 13th will be considered the final version, even if you had a correct version earlier. We will retrieve and grade the code at that time.

## Rubric

This milestone contributes to 10% of the overall project score. The score is determined by the correctness and timing of your code.

* CPU Implementation ( 4% )
* Basic GPU Implementation ( 6% )

Your grade will be determined by the number of test cases your code passes. Test cases differ in batch sizes (100, 1000, 10000). We will run each test for accuracy, but timing requirements are only evaluated for batch size 10000. There may be hidden test cases that will be used for grading.

## Appendix

The appendix is optional and not directly related to graded tasks. Feel free to read it if you want to learn more about the project.

### Checking for Errors

Within `project/src/layer/custom/new-forward.cu`, you can use the predefined error handling code to catch CUDA errors.

To catch memory errors, prepend your command with `cuda-memcheck`.
Assume we want to check memory errors on Milestone1 GPU binary,
in your `m1_gpu.slurm`, run

    /bin/bash -c "cuda-memcheck ./m1_gpu"


### JIT Caching

`nvcc`, the CUDA compiler driver, uses a two-stage compilation model. The first stage compiles source device code to PTX virtual assembly, and the second stage compiles the PTX to binary code for the target architecture. The CUDA driver can execute the second stage compilation at run time, compiling the PTX virtual assembly "Just In Time" to run it.

JIT compilation may introduce a delay during the first run of an executable. However, once compiled, the binary code is cached, allowing subsequent runs to be faster. For instance, the sum of Op Times of the reference `m1_gpu` implementation is around 120 ms on its first run, but drops to about 70 ms on following runs due to caching.

To eliminate JIT overhead, we instruct `nvcc` to generate binary code for the target architecture ahead of time. In [CMakeLists.txt](project/CMakeLists.txt), we specify the following:

```CMake
set(CMAKE_CUDA_ARCHITECTURES 86)
```

It compiles binary code directly for the sm_86 architecture (such as the A40 GPU), ensuring that JIT overhead is avoided when running jobs on Delta.

Optional reading: [CUDA Pro Tip: Understand Fat Binaries and JIT Caching](https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/)

### Specifying the Convolution Implementation

`project/src/layer/custom/cpu-new-forward.cc` and `project/src/layer/custom/new-forward.cu` contain skeleton implementations for the CPU and GPU convolutions respectively. `project/src/layer/custom/cpu-new-forward.h` and `project/src/layer/custom/gpu-new-forward.h` are the respective header files.

### Op Time and Layer Time

"Op time" refers to the time taken by `conv_forward_gpu`. "Layer time" represents the total duration of `conv_forward_gpu_prolog`, `conv_forward_gpu`, and `conv_forward_gpu_epilog` combined.

To learn more about how these metrics are computed, please see the source code [conv_cust.cc](project/src/layer/conv_cust.cc)
