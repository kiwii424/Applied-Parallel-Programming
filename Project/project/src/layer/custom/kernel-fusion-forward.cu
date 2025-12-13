#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#ifndef TILE_W
#define TILE_W 16
#endif



__global__ void matmul_conv_fused(const float *mask, const float *input, float *output,
                                  int Batch, int Map_out, int Channel, int Height, int Width, int K)
{

    const int Ho = Height - K + 1;
    const int Wo = Width  - K + 1;

    const int image_size = Ho * Wo;                 // pixels per image
    const int midDim     = Channel * K * K;         // shared K dimension
    const int num_rows   = Map_out;                 // rows of A / C
    const int num_cols   = Batch * image_size;      // cols of B / C

    // Shared-memory tiles
    __shared__ float As[TILE_W][TILE_W];            // tile of A (mask)
    __shared__ float Bs[TILE_W][TILE_W];            // tile of B (unrolled input)

    // Output matrix coordinates (row, col) in C
    const int row = blockIdx.y * TILE_W + threadIdx.y; // output feature map index m
    const int col = blockIdx.x * TILE_W + threadIdx.x; // unrolled column index

    float acc = 0.f;

    // How many tiles along the mid dimension
    const int num_tiles = (midDim + TILE_W - 1) / TILE_W;

    for (int t = 0; t < num_tiles; ++t) {
        const int kA = t * TILE_W + threadIdx.x;    // column in A
        const int kB = t * TILE_W + threadIdx.y;    // row    in B

        if (row < num_rows && kA < midDim) {
            const int ic  = kA / (K * K);
            const int rem = kA - ic * (K * K);      // avoid modulo
            const int ky  = rem / K;
            const int kx  = rem - ky * K;

            // mask layout: [Map_out, Channel, K, K]
            As[threadIdx.y][threadIdx.x] =
                mask[((row * Channel + ic) * K + ky) * K + kx];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.f;
        }

        if (col < num_cols && kB < midDim) {
            // decode column → (b, ho, wo)
            const int b  = col / image_size;
            const int hw = col - b * image_size;
            const int ho = hw / Wo;
            const int wo = hw - ho * Wo;

            // decode kB → (ic2, ky2, kx2)
            const int ic2  = kB / (K * K);
            const int rem2 = kB - ic2 * (K * K);
            const int ky2  = rem2 / K;
            const int kx2  = rem2 - ky2 * K;

            const int hi = ho + ky2;
            const int wi = wo + kx2;

            // input layout: [Batch, Channel, Height, Width]
            Bs[threadIdx.y][threadIdx.x] =
                input[((b * Channel + ic2) * Height + hi) * Width + wi];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.f;
        }

        __syncthreads();

        if (row < num_rows && col < num_cols) {
            #pragma unroll
            for (int kInner = 0; kInner < TILE_W; ++kInner) {
                acc += As[threadIdx.y][kInner] * Bs[kInner][threadIdx.x];
            }
        }

        __syncthreads();
    }

    if (row < num_rows && col < num_cols) {
        const int b  = col / image_size;
        const int hw = col - b * image_size;
        const int ho = hw / Wo;
        const int wo = hw - ho * Wo;

        // output layout: [Batch, Map_out, Ho, Wo]
        output[((b * Map_out + row) * Ho + ho) * Wo + wo] = acc;
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, 
                                                    float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, 
                                                    const int Batch, const int Map_out, const int Channel, 
                                                    const int Height, const int Width, const int K)
{

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


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask,     
                                            const int Batch, const int Map_out, const int Channel, 
                                            const int Height, const int Width, const int K)
{
    
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


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input,  
                                                    float *device_mask, const int Batch, const int Map_out, 
                                                    const int Channel, const int Height, const int Width, const int K)
{
    
    const int Ho = Height - K + 1;
    const int Wo = Width  - K + 1;
    const size_t out_bytes = (size_t)Batch * Map_out * Ho * Wo * sizeof(float);

    cudaMemcpy(host_output, device_output, out_bytes, cudaMemcpyDeviceToHost);

    
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