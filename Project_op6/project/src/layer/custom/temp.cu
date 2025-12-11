#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define THREAD_TILE 2

__global__ void matmul_conv_fused(const float *mask, const float *input, float *output,
                                  int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int H_unroll = Channel * K * K;
    const int W_unroll = Batch * Height_out * Width_out;

    __shared__ float tile_mask[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_input[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y * THREAD_TILE;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float acc[THREAD_TILE] = {0.0f};

    for (int t = 0; t < (H_unroll - 1) / TILE_WIDTH + 1; ++t) {
        int tiled_k = t * TILE_WIDTH + threadIdx.x;

        if (row < Map_out && tiled_k < H_unroll) {
            int c = tiled_k / (K * K);
            int p = (tiled_k % (K * K)) / K;
            int q = (tiled_k % (K * K)) % K;
            tile_mask[threadIdx.y * THREAD_TILE][threadIdx.x] = mask[row * (Channel * K * K) + c * (K * K) + p * K + q];
        } else {
            tile_mask[threadIdx.y * THREAD_TILE][threadIdx.x] = 0.0f;
        }

        if (row + 1 < Map_out && tiled_k < H_unroll) {
            int c = tiled_k / (K * K);
            int p = (tiled_k % (K * K)) / K;
            int q = (tiled_k % (K * K)) % K;
            tile_mask[threadIdx.y * THREAD_TILE + 1][threadIdx.x] = mask[(row + 1) * (Channel * K * K) + c * (K * K) + p * K + q];
        } else {
            tile_mask[threadIdx.y * THREAD_TILE + 1][threadIdx.x] = 0.0f;
        }

        int tiled_row = t * TILE_WIDTH + threadIdx.y * THREAD_TILE;

        if (col < W_unroll && tiled_row < H_unroll) {
            int b = col / (Height_out * Width_out);
            int hw = col % (Height_out * Width_out);
            int h_out = hw / Width_out;
            int w_out = hw % Width_out;

            int c = tiled_row / (K * K);
            int p = (tiled_row % (K * K)) / K;
            int q = (tiled_row % (K * K)) % K;

            int h = h_out + p;
            int w = w_out + q;

            tile_input[threadIdx.y * THREAD_TILE][threadIdx.x] =
                input[b * Channel * Height * Width + c * Height * Width + h * Width + w];
        } else {
            tile_input[threadIdx.y * THREAD_TILE][threadIdx.x] = 0.0f;
        }

        tiled_row++;

        if (col < W_unroll && tiled_row < H_unroll) {
            int b = col / (Height_out * Width_out);
            int hw = col % (Height_out * Width_out);
            int h_out = hw / Width_out;
            int w_out = hw % Width_out;

            int c = tiled_row / (K * K);
            int p = (tiled_row % (K * K)) / K;
            int q = (tiled_row % (K * K)) % K;

            int h = h_out + p;
            int w = w_out + q;

            tile_input[threadIdx.y * THREAD_TILE + 1][threadIdx.x] =
                input[b * Channel * Height * Width + c * Height * Width + h * Width + w];
        } else {
            tile_input[threadIdx.y * THREAD_TILE + 1][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            acc[0] += tile_mask[threadIdx.y * THREAD_TILE][i] * tile_input[i][threadIdx.x];
            acc[1] += tile_mask[threadIdx.y * THREAD_TILE + 1][i] * tile_input[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < Map_out && col < W_unroll) {
        int b = col / (Height_out * Width_out);
        int hw = col % (Height_out * Width_out);
        int h = hw / Width_out;
        int w = hw % Width_out;

        output[b * Map_out * Height_out * Width_out + row * Height_out * Width_out + h * Width_out + w] = acc[0];
    }

    row++;

    if (row < Map_out && col < W_unroll) {
        int b = col / (Height_out * Width_out);
        int hw = col % (Height_out * Width_out);
        int h = hw / Width_out;
        int w = hw % Width_out;

        output[b * Map_out * Height_out * Width_out + row * Height_out * Width_out + h * Width_out + w] = acc[1];
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask,
                                                    float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr,
                                                    const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
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

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask,
                                             const int Batch, const int Map_out, const int Channel,
                                             const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int W_unroll = Batch * Height_out * Width_out;

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH / THREAD_TILE, 1);
    dim3 dimGrid((W_unroll - 1) / TILE_WIDTH + 1,
                 (Map_out - 1) / TILE_WIDTH + 1, 1);

    matmul_conv_fused<<<dimGrid, dimBlock>>>(device_mask, device_input, device_output,
                                             Batch, Map_out, Channel, Height, Width, K);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask,
                                                    const int Batch, const int Map_out, const int Channel,
                                                    const int Height, const int Width, const int K)
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