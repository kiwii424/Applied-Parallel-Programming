#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>

#define TILE_WIDTH 16

// --------------- device kernel (FP16 with __half2) ---------------

__global__ void matmul_conv_fused_fp16(
    const float *mask,
    const float *input,
    float *output,
    int Batch, int Map_out, int Channel,
    int Height, int Width, int K)
{
    const int H_out = Height - K + 1;
    const int W_out = Width  - K + 1;
    const int H_unroll = Channel * K * K;
    const int W_unroll = Batch * H_out * W_out;

    __shared__ __half2 shMask[TILE_WIDTH][TILE_WIDTH];
    __shared__ __half2 shInput[TILE_WIDTH][TILE_WIDTH];

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row = blockIdx.y * TILE_WIDTH + ty;  // m (output channel)
    int col = blockIdx.x * TILE_WIDTH + tx;  // flattened (b, h_out, w_out)

    // accumulator in FP32
    float acc = 0.0f;

    // helper macros for indexing
    #define MASK_4D(m, c, p, q) \
        mask[(m) * (Channel * K * K) + (c) * (K * K) + (p) * K + (q)]

    #define IN_4D(b, c, h, w) \
        input[(b) * (Channel * Height * Width) + \
              (c) * (Height * Width) + \
              (h) * Width + (w)]

    // number of tiles along K*K*C dimension
    int numTiles = (H_unroll + TILE_WIDTH - 1) / TILE_WIDTH;

    // decode col -> (b, h_out, w_out)
    int b = 0, h_out = 0, w_out = 0;
    if (col < W_unroll) {
        int hw = col % (H_out * W_out);
        b      = col / (H_out * W_out);
        h_out  = hw / W_out;
        w_out  = hw % W_out;
    }

    for (int t = 0; t < numTiles; ++t) {
        int k_index_for_mask  = t * TILE_WIDTH + tx;  // column index in A
        int k_index_for_input = t * TILE_WIDTH + ty;  // row index in B

        // load mask tile
        float m_val = 0.0f;
        if (row < Map_out && k_index_for_mask < H_unroll) {
            int c_idx  = k_index_for_mask / (K * K);
            int rem    = k_index_for_mask % (K * K);
            int p_idx  = rem / K;
            int q_idx  = rem % K;
            m_val = MASK_4D(row, c_idx, p_idx, q_idx);
        }
        shMask[ty][tx] = __halves2half2(__float2half(m_val), __float2half(0.0f));

        // load input tile
        float x_val = 0.0f;
        if (col < W_unroll && k_index_for_input < H_unroll) {
            int c_idx  = k_index_for_input / (K * K);
            int rem    = k_index_for_input % (K * K);
            int p_idx  = rem / K;
            int q_idx  = rem % K;

            int h_in = h_out + p_idx;
            int w_in = w_out + q_idx;

            x_val = IN_4D(b, c_idx, h_in, w_in);
        }
        shInput[ty][tx] = __halves2half2(__float2half(x_val), __float2half(0.0f));

        __syncthreads();

        // compute partial dot product for this tile
        if (row < Map_out && col < W_unroll) {
            #pragma unroll
            for (int k = 0; k < TILE_WIDTH; ++k) {
                __half2 prod = __hmul2(shMask[ty][k], shInput[k][tx]);
                float2 prod_f = __half22float2(prod);
                acc += prod_f.x + prod_f.y;  // second lane目前是0，不影響正確性
            }
        }

        __syncthreads();
    }

    // write result
    if (row < Map_out && col < W_unroll) {
        int hw = col % (H_out * W_out);
        int h  = hw / W_out;
        int w  = hw % W_out;
        output[b * Map_out * H_out * W_out +
               row * H_out * W_out +
               h   * W_out +
               w] = acc;
    }

    #undef MASK_4D
    #undef IN_4D
}

// --------------- host interface ---------------

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
    size_t n_in     = (size_t)Batch * Channel * Height * Width;
    size_t n_out    = (size_t)Batch * Map_out * (Height - K + 1) * (Width - K + 1);
    size_t n_filter = (size_t)Map_out * Channel * K * K;

    size_t input_bytes  = n_in     * sizeof(float);
    size_t output_bytes = n_out    * sizeof(float);
    size_t mask_bytes   = n_filter * sizeof(float);

    cudaMalloc((void**)device_input_ptr,  input_bytes);
    cudaMalloc((void**)device_mask_ptr,   mask_bytes);
    cudaMalloc((void**)device_output_ptr, output_bytes);

    cudaMemcpy(*device_input_ptr, host_input, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr,  host_mask,  mask_bytes,  cudaMemcpyHostToDevice);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error in conv_forward_gpu_prolog (op_5): "
                  << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

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
    const int H_out    = Height - K + 1;
    const int W_out    = Width  - K + 1;
    const int W_unroll = Batch * H_out * W_out;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim((W_unroll + TILE_WIDTH - 1) / TILE_WIDTH,
                 (Map_out  + TILE_WIDTH - 1) / TILE_WIDTH,
                 1);

    matmul_conv_fused_fp16<<<gridDim, blockDim>>>(
        device_mask, device_input, device_output,
        Batch, Map_out, Channel, Height, Width, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA kernel launch error in conv_forward_gpu (op_5): "
                  << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

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
    size_t n_out = (size_t)Batch * Map_out * H_out * W_out;
    size_t output_bytes = n_out * sizeof(float);

    cudaMemcpy(host_output, device_output, output_bytes, cudaMemcpyDeviceToHost);

    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error in conv_forward_gpu_epilog (op_5): "
                  << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
        std::cout << "Computational capabilities: " << deviceProp.major
                  << "." << deviceProp.minor << std::endl;
        std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
        std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max block dimensions: "
                  << deviceProp.maxThreadsDim[0] << " x, "
                  << deviceProp.maxThreadsDim[1] << " y, "
                  << deviceProp.maxThreadsDim[2] << " z" << std::endl;
        std::cout << "Max grid dimensions: "
                  << deviceProp.maxGridSize[0] << " x, "
                  << deviceProp.maxGridSize[1] << " y, "
                  << deviceProp.maxGridSize[2] << " z" << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    }
}