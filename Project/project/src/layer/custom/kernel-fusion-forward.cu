#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#ifndef TILE_W
#define TILE_W 16
#endif


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

    const int Ho = Height - K + 1;
    const int Wo = Width  - K + 1;

    // Unrolled shapes: A = [Map_out, Channel*K*K], B = [Channel*K*K, Batch*Ho*Wo]
    const int A_rows = Map_out;
    const int A_cols = Channel * K * K;       // also B_rows
    const int B_cols = Batch * Ho * Wo;

    // Tile buffers for block-level GEMM
    __shared__ float As[TILE_W][TILE_W];
    __shared__ float Bs[TILE_W][TILE_W];

    // Output tile coordinates
    const int row = blockIdx.y * TILE_W + threadIdx.y;   // 0..Map_out-1
    const int col = blockIdx.x * TILE_W + threadIdx.x;   // 0..B_cols-1

    float acc = 0.f;

    // Loop over the unrolled K dimension (A_cols / B_rows)
    const int tiles = (A_cols + TILE_W - 1) / TILE_W;

    for (int t = 0; t < tiles; ++t) {
        // Global indices for this tile chunk
        const int a_col = t * TILE_W + threadIdx.x;      // column in A
        const int b_row = t * TILE_W + threadIdx.y;      // row in B

        // Load A tile: A[row, a_col] = mask[oc=row, ic/k/p/q derived from a_col]
        if (row < A_rows && a_col < A_cols) {
            const int ic  = a_col / (K * K);
            const int rem = a_col % (K * K);
            const int ky  = rem / K;
            const int kx  = rem % K;
            As[threadIdx.y][threadIdx.x] = mask[((row * Channel + ic) * K + ky) * K + kx];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.f;
        }

        // Load B tile: B[b_row, col] = input[derived from (b_row, col)]
        if (col < B_cols && b_row < A_cols) {
            // Decode col → (b, ho, wo)
            const int b   = col / (Ho * Wo);
            const int hw  = col % (Ho * Wo);
            const int ho  = hw / Wo;
            const int wo  = hw % Wo;

            // Decode b_row → (ic, ky, kx)
            const int ic2  = b_row / (K * K);
            const int rem2 = b_row % (K * K);
            const int ky2  = rem2 / K;
            const int kx2  = rem2 % K;

            const int hi = ho + ky2;
            const int wi = wo + kx2;

            Bs[threadIdx.y][threadIdx.x] =
                input[ ((b * Channel + ic2) * Height + hi) * Width + wi ];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.f;
        }

        __syncthreads();

        // Accumulate this tile’s partial product
        #pragma unroll
        for (int kInner = 0; kInner < TILE_W; ++kInner) {
            acc += As[threadIdx.y][kInner] * Bs[kInner][threadIdx.x];
        }
        __syncthreads();
    }

    // Write back acc → output[b, oc=row, ho, wo]
    if (row < A_rows && col < B_cols) {
        const int b   = col / (Ho * Wo);
        const int hw  = col % (Ho * Wo);
        const int ho  = hw / Wo;
        const int wo  = hw % Wo;

        output[ ((b * Map_out + row) * Ho + ho) * Wo + wo ] = acc;
    }

}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    const int Ho = Height - K + 1;
    const int Wo = Width  - K + 1;

    const size_t in_bytes  = (size_t)Batch * Channel * Height * Width * sizeof(float);
    const size_t w_bytes   = (size_t)Map_out * Channel * K * K * sizeof(float);
    const size_t out_bytes = (size_t)Batch * Map_out * Ho * Wo * sizeof(float);

    cudaMalloc((void**)device_input_ptr,  in_bytes);
    cudaMalloc((void**)device_mask_ptr,   w_bytes);
    cudaMalloc((void**)device_output_ptr, out_bytes);

    cudaMemcpy(*device_input_ptr,  host_input, in_bytes,  cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr,   host_mask,  w_bytes,   cudaMemcpyHostToDevice);

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Set the kernel dimensions and call the fused kernel

    const int Ho = Height - K + 1;
    const int Wo = Width  - K + 1;

    const int A_rows = Map_out;
    const int B_cols = Batch * Ho * Wo;

    dim3 block(TILE_W, TILE_W, 1);
    dim3 grid( (B_cols + TILE_W - 1) / TILE_W,
               (A_rows + TILE_W - 1) / TILE_W, 1);

    matmul_conv_fused<<<grid, block>>>(device_mask, device_input, device_output,
                                       Batch, Map_out, Channel, Height, Width, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host

    const int Ho = Height - K + 1;
    const int Wo = Width  - K + 1;
    const size_t out_bytes = (size_t)Batch * Map_out * Ho * Wo * sizeof(float);

    cudaMemcpy(host_output, device_output, out_bytes, cudaMemcpyDeviceToHost);

    
    // TODO: Free device memory
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