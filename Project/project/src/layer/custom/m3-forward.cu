#include <cmath>
#include <iostream>
#include <mma.h>
#include "gpu-new-forward.h"

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define TILE_WIDTH 16

__global__ void matmul_conv_fused(const float *mask, const float *input, float *output,
                                  int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int H_unroll = Channel * K * K;
    const int W_unroll = Batch * Height_out * Width_out;

    __shared__ __half a_tile[WMMA_M * WMMA_N];
    __shared__ __half b_tile[WMMA_M * WMMA_N];
    __shared__ float C_tile[WMMA_M * WMMA_N];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    __half filter = 0.0f;

    for (int t = 0; t < (H_unroll - 1) / TILE_WIDTH + 1; ++t) {
        int tiled_k = t * TILE_WIDTH + threadIdx.x;

        int ac = tiled_k / (K * K);
        int ap = (tiled_k % (K * K)) / K;
        int aq = (tiled_k % (K * K)) % K;

        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i += 2) {
            filter = (row + i) < Map_out && tiled_k < H_unroll;
            a_tile[(threadIdx.y + i) * TILE_WIDTH + threadIdx.x] = __float2half(mask[(row + i) * (Channel * K * K) + ac * (K * K) + ap * K + aq]) * filter;
        }

        int b = col / (Height_out * Width_out);
        int hw = col % (Height_out * Width_out);
        int h_out = hw / Width_out;
        int w_out = hw % Width_out;

        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i += 2) {
            int tiled_row = t * TILE_WIDTH + threadIdx.y + i;

            int c = tiled_row / (K * K);
            int p = (tiled_row % (K * K)) / K;
            int q = (tiled_row % (K * K)) % K;

            int h = h_out + p;
            int w = w_out + q;

            filter = col < W_unroll && tiled_row < H_unroll;

            b_tile[(threadIdx.y + i) * TILE_WIDTH + threadIdx.x] =
                __float2half(input[b * Channel * Height * Width + c * Height * Width + h * Width + w]) * filter;
        }

        __syncthreads();

        wmma::load_matrix_sync(a_frag, a_tile, WMMA_M);
        wmma::load_matrix_sync(b_frag, b_tile, WMMA_N);

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        __syncthreads();
    }

    wmma::store_matrix_sync(C_tile, acc_frag, WMMA_N, wmma::mem_row_major);

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < TILE_WIDTH; i += 2) {
        if ((row + i) < Map_out && col < W_unroll) {
            int b = col / (Height_out * Width_out);
            int hw = col % (Height_out * Width_out);
            int h = hw / Width_out;
            int w = hw % Width_out;
    
            output[b * Map_out * Height_out * Width_out + (row + i) * Height_out * Width_out + h * Width_out + w] = C_tile[(threadIdx.y + i) * WMMA_N + threadIdx.x];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    size_t input_size = Batch * Channel * Height * Width * sizeof(float);
    size_t mask_size = Map_out * Channel * K * K * sizeof(float);
    size_t output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float);

    cudaMalloc((void**)device_input_ptr, input_size);
    cudaMalloc((void**)device_mask_ptr, mask_size);
    cudaMalloc((void**)device_output_ptr, output_size);

    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int W_unroll = Batch * Height_out * Width_out;

    dim3 dimBlock(TILE_WIDTH, 2, 1);
    dim3 dimGrid((W_unroll - 1) / TILE_WIDTH + 1,
                 (Map_out - 1) / TILE_WIDTH + 1, 1);

    matmul_conv_fused<<<dimGrid, dimBlock>>>(device_mask, device_input, device_output,
                                             Batch, Map_out, Channel, Height, Width, K);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    size_t output_size = Batch * Map_out * Height_out * Width_out * sizeof(float);

    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
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