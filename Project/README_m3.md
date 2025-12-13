# Milestone 3: GPU Convolution Kernel Optimizations

Deadline: **December 5th, 2025, 8 PM CST**

| Step | Deliverables                                                               |
| ---- | -------------------------------------------------------------------------- |
| 1    | Implement multiple GPU optimizations individually                          |
| 2    | Combine one or more optimizations to achieve op times (sum) of <= **60ms** |
| 3    | Write your report and upload PDF to Gradescope                             |
| 4    | Submit your work for grading!                                              |


You will edit the following files for milestone 3.
```
project/src/layer/custom/m3-forward.cu
project/m3/*
```

**Only modify the files specifically mentioned in this document. Changes to other files will not be used for grading, and may cause unexpected errors that you will be responsible for.**

## Table of Contents
- [Create your own GPU optimizations! The real race against time.](#create-your-own-gpu-optimizations-the-real-race-against-time)
  - [Performance Analysis using Nsight-Systems and Nsight-Compute](#performance-analysis-using-nsight-systems-and-nsight-compute)
  - [Documentation on running your code](#documentation-on-running-your-code)
- [Submission Guidelines](#submission-guidelines)
  - [Code Submission Guideline through Github](#code-submission-guideline-through-github)
  - [Profiling Results Submission Guideline through Google Drive](#profiling-results-submission-guideline-through-google-drive)
  - [Milestone 3 Report Submission Guidelines through Gradescope](#milestone-3-report-submission-guidelines-through-gradescope)
- [Optimizations](#optimizations)
  - [Requirements and Hints for Implementing the Optimizations](#requirements-and-hints-for-implementing-the-optimizations)
  - [Extra credits in the project](#extra-credits-in-the-project)
- [Rubric](#rubric)

## Create your own GPU optimizations! The real race against time.

You will be optimizing your milestone 2 code. Your goal is to implement the two required optimizations (Streams and Tensor Cores) and at least **10 additional points** of optional optimizations (as seen in [optimizations](#optimizations)). Additional optimization points beyond the required total will count towards extra credits.

You will implement each optimization individually, and then select one or more and combine them in `/project/src/layer/custom/m3-forward.cu`. This file will be your final code submission, where the goal is to maximize performance.

**Individual Optimizations**

You will analyze any optimization you implement separately. Any analysis on individual optimization should be compared against your Milestone 2 Kernel Fusion or Unfused Input Unrolling, as specified in the instructions.

Op times are sometimes quite arbitrary and is not the only way to show improvement in your kernel. It is fine if an optimization is not improving the performance against the baseline,
but you have to provide your implementation in your code and sufficient profiling results in your report.

**Final Submission**

Although you are required to implement the Streams optimization, for the purpose of the final performance test, you should **disable** multiple streams and use a **single stream** in your `/project/src/layer/custom/m3-forward.cu`. This is because Op Times are not a reliable metric for evaluating multi-stream applications.

Your final submission must have correct accuracy for any batch size. Therefore, avoid any optimizations that could impact accuracy in your final submission, such as FP16. (Learn more about FP16's impact on accuracy [here](#op_5-fp16).) You may still implement FP16 as an individual optimization and it will count towards the 10 points of optional optimizations.

If you have done milestone 2 correctly, for a batch size of 10000, the sum between the first and second layer OP Times should equal about **60ms**.

In order to achieve full credit for the performance in milestone 3, your final submission must bring down the sum of the op times to **60ms** or less for a batch size of 10000. Any submissions between **60ms** and **100ms** will be given a performance grade linearly extrapolated from the performance relative to these two values.

Any submission slower than **100ms** will receive no credit for the performance test.

We will **only** run your `m3-forward.cu` file inside of `/project/src/layer/custom/` when we evaluate your performance.

All your gpu kernel calls need to take place inside `conv_forward_gpu()` for final submission.

### Performance Analysis using Nsight-Systems and Nsight-Compute

Please view the profiling staff lecture [Application Profiling with Nsight Systems & Nsight Compute](https://mediaspace.illinois.edu/media/t/1_ex8x2wep/383926342) before doing performance analysis.

Use the NVIDIA Nsight-Systems (`nsys`) and Nsight-Compute (`ncu`) and your analysis information to describe the effect that your optimizations had on the performance of your convolution. The profiling results will be used in your report.
If possible, you should try to separate the effect of each optimization in your analysis.

Please ensure that your submission includes both binary files for profiling (Nsight-Systems and Nsight-Compute) for each of your optimization. Check [submission guidelines](#submission-guidelines) for more information.

### Documentation on running your code

When profiling your optimizations, replace the `#SBATCH --constraint="projects"` with `#SBATCH --constraint="projects,perf,nvperf"` flag to run your code.

Please **do not** run your code with `#SBATCH --constraint="projects,perf,nvperf"` **if you are not actively profiling your code**. Not only will it take longer for you to run your code, it slows traffic for everyone. For basic functionality/optime check, please use the flag `#SBATCH --constraint="projects"`. This flag can be found in `./Project/m3.slurm`.

Remove `-DCMAKE_CXX_FLAGS=-pg` in `run.sh` if you haven't done this in Milestone 2.

To compile your code (do this for every change you make):

- `./run.sh build`

To run your code:

- `sbatch m3.slurm`

Checking your output:

- `m3.out` has your outputs
- `Milestone3.err` has your error outputs. Check here for seg-faults or anything similar.

## Submission Guidelines

For **Project Milestone 3**, you will need to submit your work across three platforms:

1. **GitHub**: Upload your final code for the performance test, and the code for individual optimizations.
2. **Google Drive**: Submit the output and profiling results for each individual optimization.
3. **Gradescope**: Upload the project report.

### Code Submission Guideline through Github

- Your **final** code submission (stacked optimizations or not) should be your `/project/src/layer/custom/m3-forward.cu` file.
  - This is your final submission that we will test for a combined optime of <= 60ms.
  - **Though streams is mandatory as part of your optimizations, your final submission for performance test must be done on a single stream.**
- Your individual optimizations code submission will be in the `/project/m3` folder. Look under `m3` and find the optimization folders.
  - **Each** optimization you implemented should have each own folder with the following requirements:
    - name of the folder should have the following format:`req_#` or `op_#`. (see the optimization numbers in [optimizations](#optimizations))
    - it should contain a non-stacked version of your implementation
      - a functional copy of `m3-forward.cu` with **ONLY** this implementation added on from the baseline (PM2 unfused unrolling or kernel fusion as specified in the optimization table)
      - the code file must be named `m3-forward.cu`
      - we will perform functionality checks on every individual optimization
    - feel free to add more folders if needed.
  - **You must have a folder for each optimization individually** even if you stacked all of them for your final submission.

  ``` 
  |---> /m3
      |---> /req_0
          |---> m3-forward.cu
      |---> /req_1
      |---> /op_2
      |---> /op_3
  ```

- Push your code to GitHub!
  - Only add your changes in `/project/src/layer/custom/m3-forward.cu` and `/project/m3`
- **We strongly recommend that you periodically make commits**, local or not, to ensure that you have a record of your work saved. You can always just soft reset to merge your commits. It also provides proof in case something goes wrong with submissions.


### Profiling Results Submission Guideline through Google Drive

We use Google Drive to collect profiling results because the file sizes are too large to upload to GitHub.

- Your `netid@illinois.edu` email address is linked to a Google Account known as **Google Apps @ Illinois**. If you haven't set up this account yet, please follow the instructions provided [here](https://help.uillinois.edu/TDClient/42/UIUC/Requests/ServiceDet?ID=135).
- Log in to your Google Apps @ Illinois account and go to Google Drive. Make sure to use your `@illinois.edu` Google account, not a personal Google account.
- Copy this [template folder](https://drive.google.com/drive/folders/1kVCLeyqU259bILJlajjFZTNuokZniRM9?usp=drive_link) to your Google Drive. This folder will serve as the location for submitting your individual optimization files.
- To share the folder, right-click on the copied `m3` folder, select Share, then click Share again. Grant Viewer access to the group Google Apps @ Illinois. This allows everyone with a UIUC account, including TAs, to view your submission. Do not manually share via email any of the course staff as graders may change.
  
  <img src="https://bluerose73.github.io/image-bed/ece408-fa24/granting_viewer_access.jpg" alt="granting-viewer-access" width=500>
- Look under `m3` and find the optimization folders.
- **Each** optimization you implemented should have each own folder with the following requirements:
  - name of the folder should have the following format:`req_#` or `op_#`. (see the optimization numbers in [optimizations](#optimizations))
  - it should contain the execution output and all profiling results (your outputted binary analysis files) that you included in your final report.
  - feel free to add more folders if needed.
- **You must have a folder for each optimization individually** even if you stacked all of them for your final submission.
- Include the Google Drive link to the `m3` folder on the first page of your PDF report, and provide a link to the relevant subfolder in the section for each optimization.

``` 
|---> /m3
    |---> /req_0
        |---> m3.out
        |---> analysis.ncu-rep
        |---> profile.out(optional)
        |---> analysis.nsys-rep(optional)
        |--->...( other useful profiling results)
    |---> /req_1
    |---> /op_2
    |---> /op_3
```

### Milestone 3 Report Submission Guidelines through Gradescope

As the world's best engineers and scientists, it is imperative to document our work meticulously and analyze data with scientific rigor. When analyzing statistical results from your profiling results, we recommend to take a look at this [thesis](http://impact.crhc.illinois.edu/shared/report/phd-thesis-shane-ryoo.pdf) and pay particular attention to Section 5.1 for reference and inspiration.

**We give you a report template: `ECE408_FA25_netid_m3_report.docx`.** Please use this document to get started with your report.

Follow the following steps for each GPU optimization:

| Step | For each optimization                                                                                                                                     |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | Name the optimization and corresponding number                                                                                                            |
| 2    | How does this optimization theoretically optimize your convolution kernel? Expected behavior?                                                             |
| 3    | How did you implement your code? Explain thoroughly and show code snippets. Justify the correctness of your implementation with proper profiling results. |
| 4    | Did the performance match your expectation? Explain why or why not, by analyzing profiling results.                                                       |
| 5    | Does the optimization synergize with any of other optimizations? How?                                                                                     |
| 6    | List your references used while implementing this technique. (you must mention textbook pages when relevant at the minimum)                                             |

You can access Gradescope through Canvas. Look for the assignment titled `CNN Project Milestone 3 Report`.

When Submitting:

- be sure to include any external references used during identification or development of the optimization.
- export your report as pdf and name it as `ECE408_FA25_netid_m3_report.pdf`.
- upload the report to Gradescope. During the submission process, you will be prompted to assign pages to optimization numbers. For example, page 1-2 is for `req_0`, page 3 is for `req_1`, etc.

## Optimizations

These are the list of optimizations we will consider valid for Milestone 3. To obtain full credit for Milestone 3, you must implement `req_0`, `req_1`, and a total of 10 points of optional optimizations at your discretion. Please note that these optimizations build on your work from Milestone 2 Fused or Unfused Input Unrolling as specified by the instructions, meaning you will continue to implement convolution using matrix unroll. If you would like to implement a potential optimization that is not on this list, please consult a TA or instructor beforehand to verify that the optimization is valid and we will assign it a point value. We'd love to hear about your creative ideas!

| Number    | Optimization                                                                                                                   | Baseline   | Points               |
| --------- | ------------------------------------------------------------------------------------------------------------------------------ | ---------- | -------------------- |
| **req_0** | **Using Streams to overlap computation with data transfer (required)**                                                         | Unfused Input Unrolling | -                    |
| **req_1** | **Using Tensor Cores to speed up matrix multiplication (required)**                                                            | Kernel Fusion Unrolling | -                    |
| op_0      | Weight matrix (Kernel) in constant memory                                                                                      | Kernel Fusion Unrolling | 1                    |
| op_1      | `__restrict__` keyword                                                                                                         | Kernel Fusion Unrolling | 1                    |
| op_2      | Loop unrolling                                                                                                                 | Kernel Fusion Unrolling | 1                    |
| op_3      | Sweeping various parameters to find best values (block sizes, amount of thread coarsening) -- requires tables/graphs in Report | Kernel Fusion Unrolling | 3                    |
| op_4      | Using cuBLAS for matrix multiplication                                                                                         | Unfused Input Unrolling | 3                    |
| op_5      | Fixed point (FP16) arithmetic implementation (this can modify model accuracy slightly)                                         | Kernel Fusion Unrolling | [2 or 4](#op_5-fp16) |
| op_6      | Using Joint Register and Shared Memory Tiling to speed up matrix multiplication                                                | Kernel Fusion Unrolling | 4                    |


### Requirements and Hints for Implementing the Optimizations

#### req_0 Streams

Unlike most other optimization tasks, which are based on your kernel fusion implementation, the Streams optimization builds upon your unfused input unrolling code, where three kernels - unrolling, matrix multiplication, and permutation - are launched. We made this exception because, in real-world applications, it is common to launch multiple kernels within a single stream.

In this optimization task, the goal is to overlap data transfer with kernel execution. However, the `conv_forward_gpu` function lacks access to the host memory pointers, which may complicate your implementation. To address this issue, consider one of the following solutions:

- Define additional global or static variables to store the host memory pointers.
- Do all the work in the `conv_forward_gpu_prolog` function.

To overlap kernel execution and data transfers, the host memory involved in the data transfer must be pinned memory. See [How to Overlap Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/).

#### req_1 Tensor Cores

Tensor Cores are covered in the [lecture](https://mediaspace.illinois.edu/media/t/1_cjam1djt). For this optimization, you're expected to use Tensor Cores via Warp Matrix Functions to achieve faster matrix multiplications. **Using high-level libraries such as cuBLAS will not count as using Tensor Cores.** Note that Tensor Core implementation should use **TF32 format**. If implementing with FP16, you will receive at most 5 points. **This means your accuracy should match the accuracies detailed in previous Milestones.** Refer to the following resources for guidance:

- [Programming Tensor Cores in CUDA 9](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9)
- [Warp Matrix Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-matrix-functions)

#### op_1 `__restrict__`

Please read [CUDA Pro Tip: Optimize for Pointer Aliasing](https://developer.nvidia.com/blog/cuda-pro-tip-optimize-pointer-aliasing/).

#### op_2 Loop Unrolling

Loop unrolling is an optimization technique in which a loop's iterations are expanded to reduce the overhead of loop control and potentially increase parallelism. By manually or compiler-unrolling a loop, you can often improve performance. For example,

```c
// Before unrolling
for (int i = 0; i < 16; i++) {
    sum += arr[i];
}

// After unrolling: processing 4 elements per iteration
for (int i = 0; i < 16; i += 4) {
    sum += arr[i];
    sum += arr[i + 1];
    sum += arr[i + 2];
    sum += arr[i + 3];
}
```

In the unrolled version, each loop iteration now processes four elements, reducing the number of loop control operations. This can improve performance by minimizing branching overhead and increasing instruction-level parallelism.

#### op_4 cuBLAS

The CUDA Basic Linear Algebra Subprograms (cuBLAS) library is a GPU-accelerated library that provides standard matrix and vector operations. It's optimized for NVIDIA GPUs and is widely used for efficient implementations of linear algebra routines, particularly matrix multiplication, which you'll need in this project.

Unlike most other optimization tasks, which are based on PM2 kernel fusion, the cuBLAS optimization builds upon your unfused Input Unrolling code. The reason is that it is hard to implement implicit input unrolling with cuBLAS.

For more information on using cuBLAS, see [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/index.html). Focus on reading:
- Section 1 (Introduction),
- Section 2.1.1 and 2.1.2 (Usage Basics), and
- [Section 2.7](https://docs.nvidia.com/cuda/cublas/index.html#cublas-level-3-function-reference) for matrix multiplication functions.

#### op_5 FP16

FP16 (16-bit floating point) is a data type that uses 16 bits to represent a floating-point number, allowing more efficient use of memory and computational resources at a cost of precision. CUDA provides two FP16 types, `__half` and `__half2`. `__half2` is more efficient as it packs two FP16 values into a single 32-bit register. **Using `__half2` will earn you up to 4 points, while using `__half` will earn you up to 2 points.**

Readings:
- [Mixed-Precision Programming with CUDA 8](https://developer.nvidia.com/blog/mixed-precision-programming-cuda-8/)
- [Type __half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#_CPPv46__half)
- [Half Arithmetic Functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__ARITHMETIC.html)
- [Type __half2](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2.html)
- [Half2 Arithmetic Functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF2__ARITHMETIC.html)
- [Half Precision Conversion and Data Movement](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html)

You may convert float data to and from FP16 using CUDA kernels or CPU code.

The impact of using FP16 on the final accuracy depends on how you implement it. For example, in matrix multiplication, using FP16 for multiplication while accumulating results in FP32 provides higher precision than using FP16 for both multiplication and accumulation.

In the individual optimization task, minor differences in accuracy are acceptable. However, for the final performance test, your code must produce exactly correct results. Use FP16 optimizations at your own risk here.

#### op_6 Joint Register and Shared Memory Tiling

Joint Register and Shared Memory Tiling is introduced in ECE 508. It is not a required component for ECE 408. However, if you're interested in exploring this advanced technique, the resources below can help you get started:

- [ECE 508 Spring 2023 Lecture Recording (Starting at 19 minute mark)](https://mediaspace.illinois.edu/media/t/1_tyipoq6s/287199562)
- [Lecture Slides on Joint Register and Shared Memory Tiling](https://lumetta.web.engr.illinois.edu/508/slides/lecture4.pdf)
- [The Guest Profiling Lecture](https://carlpearson.net/pdf/20200416_nsight.pdf) also discussed about this technique

### Extra credits in the project

Make sure you implement two required optimizations and additional optimizations of at least 10 points for this milestone first before considering extra credits. If you implement some optimizations incorrectly or you didn't include enough information in your report, we will not consider extra points. Additional optimization points will count towards extra credits. Each additional optimization point is worth 1%. 

## Rubric

1. Milestone 1 ( 10% )
   - CPU Implementation ( 4% )
   - Basic GPU Implementation ( 6% )
2. Milestone 2 ( 35% )
   - Unrolling ( 10% )
   - Fusion ( 10% )
   - Quiz ( 15% )
3. Milestone 3 ( 55% )
   - Overall Performance ( 25% )
     - 25 points for ≤ 60ms
     - 0 points for > 100ms
     - Linear scale between 60ms and 100ms
     - **ALL** your gpu kernel calls need to be launched inside conv_forward_gpu() for your performance submission
     - Though streams is mandatory as part of your optimizations, **your final submission for performance test must be done on a single stream.**
   - Report completeness and optimization correctness ( 30% )
     - Streams ( 10% )
     - Tensor Cores ( 10% - requires TF32; if using FP16, then at most 5% )
     - Other optimizations ( 10% ) ( 1% per optimization point )
4. Extra Credit (1% per additional optimization point)
