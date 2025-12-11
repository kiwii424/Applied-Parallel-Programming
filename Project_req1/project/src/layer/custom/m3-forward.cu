#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <mma.h>

using namespace nvcuda;

#define TILE_WIDTH 16

// use TF32 Tensor Cores with 16x16x8 tile
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

// ---------------------------------------------------------
// Fused unroll + matmul (Tensor Core) + permute
// mask  : [Map_out, Channel, K, K]
// input : [Batch, Channel, Height, Width]
// output: [Batch, Map_out, H_out, W_out]
// ---------------------------------------------------------
__global__ void matmul_conv_fused(const float *mask, const float *input, float *output,
                                  int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    /*
    TODO: Modify this function to implement the fused unroll-matmul-permute kernel.
    
    Function parameter definitions:
    mask - convolution kernel
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = Height - K + 1;
    const int W_out = Width  - K + 1;

    const int KK        = K * K;
    const int H_unroll  = Channel * KK;                 // K dimension in GEMM
    const int W_unroll  = Batch * H_out * W_out;        // N dimension in GEMM

    const int numARows    = Map_out;                    // M
    const int numAColumns = H_unroll;                   // K
    const int numBRows    = H_unroll;                   // K
    const int numBColumns = W_unroll;                   // N

    const int numCRows    = numARows;
    const int numCColumns = numBColumns;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // global C index (row: output channel, col: flattened (b, h_out, w_out))
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    // macros for indexing original 4D tensors
    #define IN4D(b, c, h, w) \
        input[(b) * (Channel * Height * Width) + (c) * (Height * Width) + (h) * Width + (w)]

    #define MASK4D(m, c, p, q) \
        mask[(m) * (Channel * K * K) + (c) * (K * K) + (p) * K + (q)]

    // shared memory tiles: 16x16 for A/B, then pack into 16x8 pieces for WMMA
    __shared__ float shA[TILE_WIDTH * TILE_WIDTH];
    __shared__ float shB[TILE_WIDTH * TILE_WIDTH];

    __shared__ float shA0[WMMA_M * WMMA_K];    // first 8 columns of A-tile
    __shared__ float shA1[WMMA_M * WMMA_K];    // second 8 columns
    __shared__ float shB0[WMMA_K * WMMA_N];    // first 8 rows of B-tile
    __shared__ float shB1[WMMA_K * WMMA_N];    // second 8 rows

    __shared__ float shC[WMMA_M * WMMA_N];     // final 16x16 C tile

    // WMMA fragments (TF32 inputs, FP32 accumulator)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // loop over K dimension in chunks of TILE_WIDTH (16)
    const int numTilesK = (H_unroll - 1) / TILE_WIDTH + 1;

    for (int t = 0; t < numTilesK; ++t) {
        int k_col = t * TILE_WIDTH + tx;    // column in A / row in B

        // -----------------------------
        // load mask tile into shA
        // -----------------------------
        if (row < numARows && k_col < numAColumns) {
            int idx = k_col;
            int c  = idx / KK;
            int rem = idx - c * KK;
            int p  = rem / K;
            int q  = rem - p * K;

            shA[ty * TILE_WIDTH + tx] = MASK4D(row, c, p, q);
        } else {
            shA[ty * TILE_WIDTH + tx] = 0.0f;
        }

        // -----------------------------
        // load input tile into shB
        // -----------------------------
        int k_row = t * TILE_WIDTH + ty;
        if (col < numBColumns && k_row < numBRows) {
            int b  = col / (H_out * W_out);
            int hw = col % (H_out * W_out);
            int h_out = hw / W_out;
            int w_out = hw - h_out * W_out;

            int idx = k_row;
            int c   = idx / KK;
            int rem = idx - c * KK;
            int p   = rem / K;
            int q   = rem - p * K;

            int h_in = h_out + p;
            int w_in = w_out + q;

            shB[ty * TILE_WIDTH + tx] = IN4D(b, c, h_in, w_in);
        } else {
            shB[ty * TILE_WIDTH + tx] = 0.0f;
        }

        __syncthreads();

        // ----------------------------------------------------
        // pack 16x16 A/B tiles into two 16x8 + two 8x16 tiles
        // so that each wmna::mma_sync sees a 16x8 * 8x16
        // ----------------------------------------------------
        // pack A: shA -> shA0 (cols 0..7) and shA1 (cols 8..15)
        if (tx < WMMA_K) {
            // first half columns for A0
            shA0[ty * WMMA_K + tx] = shA[ty * TILE_WIDTH + tx];
            // second half columns for A1
            shA1[ty * WMMA_K + tx] = shA[ty * TILE_WIDTH + (tx + WMMA_K)];
        }

        // pack B: shB -> shB0 (rows 0..7) and shB1 (rows 8..15)
        if (ty < WMMA_K) {
            shB0[ty * WMMA_N + tx] = shB[ty * TILE_WIDTH + tx];
        } else {
            int ty2 = ty - WMMA_K;  // 8..15 -> 0..7
            shB1[ty2 * WMMA_N + tx] = shB[ty * TILE_WIDTH + tx];
        }

        __syncthreads();

        // First K-slice: A0 (16x8) * B0 (8x16)
        wmma::load_matrix_sync(a_frag, shA0, WMMA_K);
        wmma::load_matrix_sync(b_frag, shB0, WMMA_N);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        // Second K-slice: A1 (16x8) * B1 (8x16)
        wmma::load_matrix_sync(a_frag, shA1, WMMA_K);
        wmma::load_matrix_sync(b_frag, shB1, WMMA_N);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        __syncthreads();
    }

    // store 16x16 C tile back to shared memory, row-major
    wmma::store_matrix_sync(shC, acc_frag, WMMA_N, wmma::mem_row_major);
    __syncthreads();

    // write results from shC to global output in NCHW layout
    if (row < numCRows && col < numCColumns) {
        int b  = col / (H_out * W_out);
        int hw = col % (H_out * W_out);
        int h  = hw / W_out;
        int w  = hw - h * W_out;

        float val = shC[ty * WMMA_N + tx];
        output[b * Map_out * H_out * W_out +
               row * H_out * W_out +
               h * W_out + w] = val;
    }

    #undef IN4D
    #undef MASK4D
}

// ---------------------------------------------------------
// Prolog: allocate device memory and copy host data
// ---------------------------------------------------------
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask,
                                                    float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr,
                                                    const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU

    const int H_out = Height - K + 1;
    const int W_out = Width  - K + 1;

    size_t input_size  = static_cast<size_t>(Batch) * Channel * Height * Width * sizeof(float);
    size_t mask_size   = static_cast<size_t>(Map_out) * Channel * K * K * sizeof(float);
    size_t output_size = static_cast<size_t>(Batch) * Map_out * H_out * W_out * sizeof(float);

    cudaMalloc((void**)device_input_ptr,  input_size);
    cudaMalloc((void**)device_mask_ptr,   mask_size);
    cudaMalloc((void**)device_output_ptr, output_size);

    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr,  host_mask,  mask_size,  cudaMemcpyHostToDevice);

    // We pass double pointers for you to initialize the relevant device pointers,
    // which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
}

// ---------------------------------------------------------
// Main conv: set launch configuration and call fused kernel
// ---------------------------------------------------------
__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask,
                                             const int Batch, const int Map_out, const int Channel,
                                             const int Height, const int Width, const int K)
{
    // TODO: Set the kernel dimensions and call the fused kernel

    const int H_out = Height - K + 1;
    const int W_out = Width  - K + 1;

    const int W_unroll = Batch * H_out * W_out;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim((W_unroll - 1) / TILE_WIDTH + 1,
                 (Map_out  - 1) / TILE_WIDTH + 1,
                 1);

    matmul_conv_fused<<<gridDim, blockDim>>>(
        device_mask, device_input, device_output,
        Batch, Map_out, Channel, Height, Width, K);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA kernel launch error (req_1 Tensor Core): "
                  << cudaGetErrorString(error) << std::endl;
        std::exit(-1);
    }
}

// ---------------------------------------------------------
// Epilog: copy result back and free device memory
// ---------------------------------------------------------
__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask,
                                                    const int Batch, const int Map_out, const int Channel,
                                                    const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host

    const int H_out = Height - K + 1;
    const int W_out = Width  - K + 1;
    size_t output_size = static_cast<size_t>(Batch) * Map_out * H_out * W_out * sizeof(float);

    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);

    // TODO: Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}

// ---------------------------------------------------------
// Helper: print device properties
// ---------------------------------------------------------
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