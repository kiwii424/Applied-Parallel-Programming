#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH  16
#define ROW_TILE    2   


__global__ void matmul_conv_fused(const float *mask, const float *input, float *output,
                                    int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    const int H_out = Height - K + 1;
    const int W_out = Width  - K + 1;
    const int H_unroll = Channel * K * K;
    const int W_unroll = Batch * H_out * W_out;

    __shared__ float shMask[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shInput[TILE_WIDTH][TILE_WIDTH];

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // global row/col for the first row this thread is responsible for
    int baseRow = blockIdx.y * TILE_WIDTH + ty * ROW_TILE;
    int col     = blockIdx.x * TILE_WIDTH + tx;

    // each thread keeps ROW_TILE partial sums in registers
    float acc[ROW_TILE];
    #pragma unroll
    for (int r = 0; r < ROW_TILE; ++r) {
        acc[r] = 0.0f;
    }

    #define MASK_4D(m, c, p, q) mask[(m) * (Channel * K * K) + (c) * (K * K) + (p) * K + (q)]
    #define IN_4D(b, c, h, w) input[(b) * (Channel * Height * Width) + (c) * (Height * Width) + (h) * Width + (w)]

    int numTiles = (H_unroll + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; ++t) {
        int k_col = t * TILE_WIDTH + tx;                
        int k_row_base = t * TILE_WIDTH + ty * ROW_TILE; 

        // load TILE of mask: ROW_TILE rows per thread
        #pragma unroll
        for (int r = 0; r < ROW_TILE; ++r) {
            int gRow = baseRow + r;
            int shRow = ty * ROW_TILE + r;

            if (gRow < Map_out && k_col < H_unroll) {
                int c  = k_col / (K * K);
                int kk = k_col % (K * K);
                int p  = kk / K;
                int q  = kk % K;
                shMask[shRow][tx] = MASK_4D(gRow, c, p, q);
            } else {
                shMask[shRow][tx] = 0.0f;
            }
        }

        // load TILE of input: also ROW_TILE rows per thread
        #pragma unroll
        for (int r = 0; r < ROW_TILE; ++r) {
            int k_row = k_row_base + r;
            int shRow = ty * ROW_TILE + r;

            if (col < W_unroll && k_row < H_unroll) {
                int b    = col / (H_out * W_out);
                int hw   = col % (H_out * W_out);
                int h_o  = hw / W_out;
                int w_o  = hw % W_out;

                int c  = k_row / (K * K);
                int kk = k_row % (K * K);
                int p  = kk / K;
                int q  = kk % K;

                int h = h_o + p;
                int w = w_o + q;

                shInput[shRow][tx] = IN_4D(b, c, h, w);
            } else {
                shInput[shRow][tx] = 0.0f;
            }
        }

        __syncthreads();

        // compute partial sums from this tile
        for (int k = 0; k < TILE_WIDTH; ++k) {
            float in_val = shInput[k][tx];
            #pragma unroll
            for (int r = 0; r < ROW_TILE; ++r) {
                int shRow = ty * ROW_TILE + r;
                acc[r] += shMask[shRow][k] * in_val;
            }
        }

        __syncthreads();
    }

    // write back both rows this thread is responsible for
    #pragma unroll
    for (int r = 0; r < ROW_TILE; ++r) {
        int gRow = baseRow + r;
        if (gRow < Map_out && col < W_unroll) {
            int b  = col / (H_out * W_out);
            int hw = col % (H_out * W_out);
            int h  = hw / W_out;
            int w  = hw % W_out;

            output[b * Map_out * H_out * W_out + gRow * H_out * W_out + h * W_out + w] = acc[r];
        }
    }

    #undef MASK_4D
    #undef IN_4D
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, 
                                                    float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, 
                                                    const int Batch, const int Map_out, const int Channel, 
                                                    const int Height, const int Width, const int K)
{
    size_t num_in   = (size_t)Batch * Channel * Height * Width;
    size_t num_out  = (size_t)Batch * Map_out * (Height - K + 1) * (Width - K + 1);
    size_t num_mask = (size_t)Map_out * Channel * K * K;

    size_t input_bytes  = num_in   * sizeof(float);
    size_t output_bytes = num_out  * sizeof(float);
    size_t mask_bytes   = num_mask * sizeof(float);

    cudaMalloc((void**)device_input_ptr,  input_bytes);
    cudaMalloc((void**)device_mask_ptr,   mask_bytes);
    cudaMalloc((void**)device_output_ptr, output_bytes);

    cudaMemcpy(*device_input_ptr, host_input, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr,  host_mask,  mask_bytes,  cudaMemcpyHostToDevice);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, 
                                            const int Batch, const int Map_out, const int Channel, const int Height, 
                                            const int Width, const int K)
{
    const int H_out = Height - K + 1;
    const int W_out = Width  - K + 1;
    const int W_unroll = Batch * H_out * W_out;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH / ROW_TILE, 1);
    dim3 gridDim((W_unroll + TILE_WIDTH - 1) / TILE_WIDTH,
                 (Map_out  + TILE_WIDTH - 1) / TILE_WIDTH,
                 1);

    matmul_conv_fused<<<gridDim, blockDim>>>(
        device_mask, device_input, device_output,
        Batch, Map_out, Channel, Height, Width, K);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, 
                                                    float *device_mask, const int Batch, const int Map_out, 
                                                    const int Channel, const int Height, const int Width, const int K)
{
    const int H_out = Height - K + 1;
    const int W_out = Width  - K + 1;
    size_t num_out = (size_t)Batch * Map_out * H_out * W_out;
    size_t output_bytes = num_out * sizeof(float);

    cudaMemcpy(host_output, device_output, output_bytes, cudaMemcpyDeviceToHost);

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