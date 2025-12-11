#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "matmul.h"
#include <cublas_v2.h>

#define PERMUTE_BLOCK_SIZE 256
#define UNROLL_THREADS 1024
#define task "op_4"

// =======================================
// Input unrolling kernel
// [Batch, Channel, Height, Width]
//   -> [Channel*K*K, Batch * H_out * W_out]
// =======================================
__global__ void unroll_input_kernel_op4(const float *input,
                                        float *unrolled,
                                        const int Batch,
                                        const int Channel,
                                        const int Height,
                                        const int Width,
                                        const int K)
{
    const size_t H_out = Height - K + 1;
    const size_t W_out = Width  - K + 1;
    const size_t cols_per_img = H_out * W_out;   // 每張圖在 unrolled 中的 column 數

    // blockIdx.y 對應 batch index
    const size_t b = blockIdx.y;
    if (b >= (size_t)Batch) return;

    // t 掃過 Channel * H_out * W_out
    const size_t t = blockIdx.x * UNROLL_THREADS + threadIdx.x;

    #define IN4D(i3, i2, i1, i0) \
        input[(i3) * (Channel * Height * Width) + \
              (i2) * (Height * Width) + \
              (i1) * Width + (i0)]

    if (t < (size_t)Channel * cols_per_img) {
        const size_t c      = t / cols_per_img;
        const size_t s      = t % cols_per_img;
        const size_t h_out  = s / W_out;
        const size_t w_out  = s % W_out;
        const size_t col_id = b * cols_per_img + h_out * W_out + w_out;  // column in unrolled

        const size_t KK = K * K;
        const size_t row_base = c * KK;  // row offset for this channel

        // 展開 KxK 視窗
        for (int p = 0; p < K; ++p) {
            for (int q = 0; q < K; ++q) {
                const size_t row_id = row_base + p * K + q;
                const int h_in = (int)h_out + p;
                const int w_in = (int)w_out + q;

                // 邊界保護 依照參考版本 B 的寫法
                if (h_in < Height && w_in < Width) {
                    unrolled[row_id * (Batch * cols_per_img) + col_id] =
                        IN4D(b, c, h_in, w_in);
                }
            }
        }
    }

    #undef IN4D
}

// =======================================
// Permutation kernel
// GEMM 輸出視作 [Map_out, Batch * image_size] row-major
// 轉成 [Batch, Map_out, H_out, W_out]
// =======================================
__global__ void permute_output_kernel_op4(const float *input,
                                          float *output,
                                          int Map_out,
                                          int Batch,
                                          int image_size)
{
    const size_t b = blockIdx.y;
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < (size_t)Batch && x < (size_t)image_size) {
        for (int m = 0; m < Map_out; ++m) {
            output[b * Map_out * image_size + m * image_size + x] =
                input[m * Batch * image_size + b * image_size + x];
        }
    }
}

// =======================================
// Prolog: allocate and copy to device
// =======================================
__host__ void GPUInterface::conv_forward_gpu_prolog(
    const float *host_output,
    const float *host_input,
    const float *host_mask,
    float **device_output_ptr,
    float **device_input_ptr,
    float **device_mask_ptr,
    const int Batch,
    const int Map_out,
    const int Channel,
    const int Height,
    const int Width,
    const int K)
{

    const int H_out = Height - K + 1;
    const int W_out = Width  - K + 1;

    size_t input_bytes  = (size_t)Batch * Channel * Height * Width * sizeof(float);
    size_t mask_bytes   = (size_t)Map_out * Channel * K * K * sizeof(float);
    size_t output_bytes = (size_t)Batch * Map_out * H_out * W_out * sizeof(float);

    cudaMalloc((void**)device_input_ptr,  input_bytes);
    cudaMalloc((void**)device_mask_ptr,   mask_bytes);
    cudaMalloc((void**)device_output_ptr, output_bytes);

    cudaMemcpy(*device_input_ptr, host_input, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr,  host_mask,  mask_bytes,  cudaMemcpyHostToDevice);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error in prolog: "
                  << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

// =======================================
// Main conv: unrolling + cuBLAS GEMM
// =======================================
__host__ void GPUInterface::conv_forward_gpu(
    float *device_output,
    const float *device_input,
    const float *device_mask,
    const int Batch,
    const int Map_out,
    const int Channel,
    const int Height,
    const int Width,
    const int K)
{
    const size_t H_out = Height - K + 1;
    const size_t W_out = Width  - K + 1;
    const size_t H_unrolled = Channel * K * K;              // rows
    const size_t W_unrolled = Batch * H_out * W_out;        // cols

    // 分配 unrolled input 與 GEMM output
    float *unrolled_matrix = nullptr;
    float *gemm_output     = nullptr;

    cudaMalloc((void**)&unrolled_matrix, H_unrolled * W_unrolled * sizeof(float));
    cudaMalloc((void**)&gemm_output,     (size_t)Batch * Map_out * H_out * W_out * sizeof(float));

    // 1. 呼叫 unrolling kernel
    {
        size_t work_per_image = (size_t)Channel * H_out * W_out;
        size_t grid_x = (work_per_image + UNROLL_THREADS - 1) / UNROLL_THREADS;
        dim3 grid_dim((unsigned int)grid_x, (unsigned int)Batch, 1);
        dim3 block_dim(UNROLL_THREADS, 1, 1);

        unroll_input_kernel_op4<<<grid_dim, block_dim>>>(
            device_input,
            unrolled_matrix,
            Batch,
            Channel,
            Height,
            Width,
            K
        );
    }

    // 2. cuBLAS GEMM
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS create handle failed in " << task << std::endl;
        cudaFree(gemm_output);
        cudaFree(unrolled_matrix);
        exit(-1);
    }

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // 在 column-major 視角下
    // A: unrolled_matrix, 維度 W_unrolled x H_unrolled
    // B: device_mask,     維度 H_unrolled x Map_out
    // C: gemm_output,     維度 W_unrolled x Map_out
    // 所以 m = W_unrolled, n = Map_out, k = H_unrolled
    stat = cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        (int)W_unrolled,
        Map_out,
        (int)H_unrolled,
        &alpha,
        unrolled_matrix,
        (int)W_unrolled,
        device_mask,
        (int)H_unrolled,
        &beta,
        gemm_output,
        (int)W_unrolled
    );

    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasSgemm failed in " << task << std::endl;
        cublasDestroy(handle);
        cudaFree(gemm_output);
        cudaFree(unrolled_matrix);
        exit(-1);
    }

    cublasDestroy(handle);

    // 3. permute 回 [Batch, Map_out, H_out, W_out]
    const size_t image_size = H_out * W_out;
    dim3 perm_grid_dim(
        (unsigned int)((image_size + PERMUTE_BLOCK_SIZE - 1) / PERMUTE_BLOCK_SIZE),
        (unsigned int)Batch,
        1
    );

    permute_output_kernel_op4<<<perm_grid_dim, PERMUTE_BLOCK_SIZE>>>(
        gemm_output,
        device_output,
        Map_out,
        Batch,
        (int)image_size
    );

    cudaFree(gemm_output);
    cudaFree(unrolled_matrix);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error in conv_forward_gpu: "
                  << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

// =======================================
// Epilog: copy back and free
// =======================================
__host__ void GPUInterface::conv_forward_gpu_epilog(
    float *host_output,
    float *device_output,
    float *device_input,
    float *device_mask,
    const int Batch,
    const int Map_out,
    const int Channel,
    const int Height,
    const int Width,
    const int K)
{
    const int H_out = Height - K + 1;
    const int W_out = Width  - K + 1;

    size_t output_bytes = (size_t)Batch * Map_out * H_out * W_out * sizeof(float);
    cudaMemcpy(host_output, device_output, output_bytes, cudaMemcpyDeviceToHost);

    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error in epilog: "
                  << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

// =======================================
// Device info
// =======================================
__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "
                 <<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "
                 <<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "
                 <<deviceProp.maxThreadsDim[0]<<" x, "
                 <<deviceProp.maxThreadsDim[1]<<" y, "
                 <<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "
                 <<deviceProp.maxGridSize[0]<<" x, "
                 <<deviceProp.maxGridSize[1]<<" y, "
                 <<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}