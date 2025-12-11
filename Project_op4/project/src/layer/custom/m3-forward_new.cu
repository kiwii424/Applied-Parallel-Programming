#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "matmul.h"
#include <cublas_v2.h>

#define PERMUTE_BLOCK_SIZE 256
#define MAX_NUM_THREADS 1024

__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K) {
    const size_t Height_out = Height - K + 1;
    const size_t Width_out = Width - K + 1;
    size_t W_unroll = Height_out * Width_out;

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

    size_t t = blockIdx.x * MAX_NUM_THREADS + threadIdx.x;
    size_t batch = blockIdx.y;
    size_t c, s, h_out, w_out, h_unroll, w_base, w_unroll;

    if (t < Channel * W_unroll) {
        c = t / W_unroll;
        s = t % W_unroll;
        h_out = s / Width_out;
        w_out = s % Width_out;
        h_unroll = h_out * Width_out + w_out;
        w_base = c * K * K;

        for (size_t p = 0; p < K; p++) {
            for (size_t q = 0; q < K; q++) {
                w_unroll = w_base + p * K + q;

                if (h_out + p < Height && w_out + q < Width) {
                    output[w_unroll * Batch * W_unroll + batch * W_unroll + h_unroll] = in_4d(batch, c, h_out + p, w_out + q);
                }
            }
        }
    }

    #undef in_4d
}

__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int image_size) {
    size_t b = blockIdx.y;
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < image_size) {
        for (size_t m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}

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
    const int K
) {
    size_t output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float);
    size_t input_size = Batch * Channel * Height * Width * sizeof(float);
    size_t mask_size = Map_out * Channel * K * K * sizeof(float);

    cudaMalloc((void**)device_output_ptr, output_size);
    cudaMalloc((void**)device_input_ptr, input_size);
    cudaMalloc((void**)device_mask_ptr, mask_size);

    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const size_t Height_out = Height - K + 1;
    const size_t Width_out = Width - K + 1;
    const size_t Height_unrolled = Channel * K * K;
    const size_t Width_unrolled = Batch * Height_out * Width_out;

    float *unrolled_matrix;
    float *matmul_output;
    cudaMalloc((void**)&unrolled_matrix, (size_t) Height_unrolled * Width_unrolled * sizeof(float));
    cudaMalloc((void**)&matmul_output, (Batch * Map_out * Height_out * Width_out) * sizeof(float));

    float *dummy_B;
    cudaMalloc((void**)&dummy_B, Batch * Map_out * Height_out * Width_out * sizeof(float));
    cudaMemset(dummy_B, 0, Batch * Map_out * Height_out * Width_out * sizeof(float));

    size_t gridDimX = ceil(Channel * Height_out * Width_out * 1.0 / MAX_NUM_THREADS);
    dim3 unrolling_kernel_grid_dim(gridDimX, Batch, 1);
    dim3 unrolling_kernel_block_dim(MAX_NUM_THREADS, 1, 1);
    matrix_unrolling_kernel<<<unrolling_kernel_grid_dim, unrolling_kernel_block_dim>>>(
        device_input, unrolled_matrix, Batch, Channel, Height, Width, K
    );

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        Width_unrolled,
        Map_out,
        Height_unrolled,
        &alpha,
        unrolled_matrix, Width_unrolled,
        device_mask, Height_unrolled,
        &beta,
        matmul_output, Width_unrolled
    );

    const size_t out_image_size = Height_out * Width_out;
    dim3 permute_kernel_grid_dim((out_image_size - 1) / PERMUTE_BLOCK_SIZE + 1, Batch, 1);
    matrix_permute_kernel<<<permute_kernel_grid_dim, PERMUTE_BLOCK_SIZE>>>(
        matmul_output, device_output, Map_out, Batch, out_image_size
    );

    cudaFree(matmul_output);
    cudaFree(unrolled_matrix);
    cublasDestroy(handle);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    size_t output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float);
    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}