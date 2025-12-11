#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
// upper bound of Map_out * Channel * K * K
#define MAX_CONST_MASK_ELEMS 15000

// convolution kernel weights in constant memory
__constant__ float const_mask[MAX_CONST_MASK_ELEMS];


// ---------------------------------------------------------
// Fused unroll + matmul + permute
// mask  : stored in const_mask as [Map_out, Channel, K, K]
// input : [Batch, Channel, Height, Width]
// output: [Batch, Map_out, H_out, W_out]
// ---------------------------------------------------------
__global__ void matmul_conv_fused(const float *mask, const float *input, float *output,
                                  int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    /*
    TODO: Modify this function to implement the fused unroll-matmul-permute kernel.
    
    Function parameter definitions:
    mask - convolution kernel   (not used directly, weights are in const_mask)
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)

    dim3 DimGrid(ceil(Map_out / TILE_WIDTH), ceil((Batch * H_out * W_out) / TILE_WIDTH), 1);
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    */

    // shared tiles for A (mask) and B (unrolled input)
    __shared__ float tile_mask[TILE_WIDTH][TILE_WIDTH];   // A tile
    __shared__ float tile_input[TILE_WIDTH][TILE_WIDTH];  // B tile

    const int H_out = Height - K + 1;
    const int W_out = Width  - K + 1;
    const int KK    = K * K;

    const int img_size = H_out * W_out;

    const int numARows    = Map_out;
    const int numAColumns = Channel * KK;          // K dimension
    const int numBRows    = numAColumns;
    const int numBColumns = Batch * img_size;      // N dimension

    const int numCRows    = numARows;
    const int numCColumns = numBColumns;

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // input  layout: [Batch, Channel, Height, Width]
    // mask   layout: [Map_out, Channel, K, K] (but stored in const_mask)
    // output layout: [Batch, Map_out, H_out, W_out]
    #define IN4D(b, c, h, w) \
        input[(b) * (Channel * Height * Width) + \
              (c) * (Height * Width) + \
              (h) * Width + (w)]

    #define MASK4D(m, c, p, q) \
        const_mask[(m) * (Channel * K * K) + \
                   (c) * (K * K) + \
                   (p) * K + (q)]

    // global C index (row, col) in the GEMM view
    int row = by * TILE_WIDTH + ty;    // corresponds to output channel m
    int col = bx * TILE_WIDTH + tx;    // corresponds to flattened (b, h_out, w_out)

    int b  = (col / img_size);
    int hw = (col % img_size);
    int h0 = (hw / W_out);
    int w0 = (hw % W_out);

    float acc = 0.0f;

    const int numTiles = (numAColumns + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; ++t) {
        int k_base = t * TILE_WIDTH;

        // load A tile from const_mask into shared memory
        if (row < numARows && (k_base + tx) < numAColumns) {
            int idx = k_base + tx;
            int c   = idx / KK;
            int rem = idx - c * KK;
            int p   = rem / K;
            int q   = rem - p * K;

            tile_mask[ty][tx] = MASK4D(row, c, p, q);
        } else {
            tile_mask[ty][tx] = 0.0f;
        }

        // load B tile from input into shared memory
        if (col < numBColumns && (k_base + ty) < numBRows) {
            int idx = k_base + ty;
            int c   = idx / KK;
            int rem = idx - c * KK;
            int p   = rem / K;
            int q   = rem - p * K;

            int h = h0 + p;
            int w = w0 + q;

            tile_input[ty][tx] = IN4D(b, c, h, w);
        } else {
            tile_input[ty][tx] = 0.0f;
        }

        __syncthreads();

        if (row < numCRows && col < numCColumns) {
            // standard tiled matmul inner product
            for (int kInner = 0; kInner < TILE_WIDTH; ++kInner) {
                acc += tile_mask[ty][kInner] * tile_input[kInner][tx];
            }
        }

        __syncthreads();
    }

    if (row < numCRows && col < numCColumns) {
        int b_out  = col / img_size;
        int hw_out = col % img_size;
        int h_out  = hw_out / W_out;
        int w_out  = hw_out % W_out;

        output[b_out * Map_out * H_out * W_out +
               row    * H_out * W_out +
               h_out  * W_out + w_out] = acc;
    }

    #undef IN4D
    #undef MASK4D
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask,
                                                    float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr,
                                                    const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU

    const int H_out = Height - K + 1;
    const int W_out = Width  - K + 1;

    size_t input_size  = static_cast<size_t>(Batch) * Channel * Height * Width * sizeof(float);
    size_t mask_elems  = static_cast<size_t>(Map_out) * Channel * K * K;
    size_t mask_size   = mask_elems * sizeof(float);
    size_t output_size = static_cast<size_t>(Batch) * Map_out * H_out * W_out * sizeof(float);

    // basic sanity check for constant memory capacity
    if (mask_elems > MAX_CONST_MASK_ELEMS) {
        std::cerr << "Error: mask size exceeds constant memory capacity in op_0" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    cudaMalloc((void**)device_input_ptr,  input_size);
    cudaMalloc((void**)device_output_ptr, output_size);
    // allocate a dummy device mask pointer (not used, but kept valid for interface)
    cudaMalloc((void**)device_mask_ptr,   mask_size);

    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);

    // copy kernel weights into constant memory
    cudaError_t err = cudaMemcpyToSymbol(const_mask, host_mask, mask_size, 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpyToSymbol(const_mask) failed: "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // We pass double pointers for you to initialize the relevant device pointers,
    // which are passed to the other two functions.

    // Useful snippet for error checking (leave commented if not needed)
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask,
                                             const int Batch, const int Map_out, const int Channel,
                                             const int Height, const int Width, const int K)
{
    // TODO: Set the kernel dimensions and call the fused kernel

    const int H_out = Height - K + 1;
    const int W_out = Width  - K + 1;

    const int W_unroll = Batch * H_out * W_out;

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((W_unroll - 1) / TILE_WIDTH + 1,
                 (Map_out  - 1) / TILE_WIDTH + 1,
                 1);

    matmul_conv_fused<<<dimGrid, dimBlock>>>(
        device_mask, device_input, device_output,
        Batch, Map_out, Channel, Height, Width, K);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA kernel launch error (op_0 const mask): "
                  << cudaGetErrorString(error) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}


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