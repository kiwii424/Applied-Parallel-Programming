#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "matmul.h"

#define PERMUTE_BLOCK_SIZE 256
#define MAX_NUM_THREADS 1024

__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K) {
    /*
    Modify this function to implement the input matrix unrolling kernel.

    Function paramter definitions:
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

    // TODO: Insert your input matrix unrolling kernel code here
   

    // row = c*K*K + p*K + q
    // col = b*(H_out*W_out) + h_out*W_out + w_out
    const size_t num_spatial = (size_t)Height_out * Width_out;
    const size_t num_cols    = (size_t)Batch * num_spatial;  // Width_unrolled

    // index from block/thread
    const int b = blockIdx.z;    // batch index
    const int c = blockIdx.y;    // channel index

    const size_t s = blockIdx.x * blockDim.x + threadIdx.x;  // spatial index
    if (b >= Batch || c >= Channel || s >= num_spatial) {
        return;
    }

    const int h_out = s / Width_out;
    const int w_out = s % Width_out;

    // Column index in unrolled matrix
    const size_t col = (size_t)b * num_spatial + s;

    // Base row offset for this channel
    const size_t row_base = (size_t)c * K * K;

    for (int p = 0; p < K; ++p) {
        const int h_in = h_out + p;
        for (int q = 0; q < K; ++q) {
            const int w_in = w_out + q;

            const size_t row = row_base + (size_t)p * K + q;
            output[row * num_cols + col] = in_4d(b, c, h_in, w_in);
        }
    }


    #undef in_4d
}


// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
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
    // TODO: Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking

    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;

    const size_t num_output = (size_t)Batch * Map_out * Height_out * Width_out;
    const size_t num_input  = (size_t)Batch * Channel * Height * Width;
    const size_t num_mask   = (size_t)Map_out * Channel * K * K;

    const size_t bytes_output = num_output * sizeof(float);
    const size_t bytes_input  = num_input  * sizeof(float);
    const size_t bytes_mask   = num_mask   * sizeof(float);

    cudaMalloc((void**)device_output_ptr, bytes_output);
    cudaMalloc((void**)device_input_ptr,  bytes_input);
    cudaMalloc((void**)device_mask_ptr,   bytes_mask);

    cudaMemcpy(*device_input_ptr, host_input, bytes_input, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr,  host_mask,  bytes_mask,  cudaMemcpyHostToDevice);

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

    float *unrolled_matrix;  // Pointer to device memory for storing the unrolled matrix
    float *matmul_output;    // Pointer to device memory for storing the result of matrix multiplication
    cudaMalloc((void**)&unrolled_matrix, (size_t) Height_unrolled * Width_unrolled * sizeof(float));
    cudaMalloc((void**)&matmul_output, (Batch * Map_out * Height_out * Width_out) * sizeof(float));

    // TODO: Set the kernel dimensions and call the matrix unrolling kernel.

    // size_t gridDimX = ceil(Channel * Height_out * Width_out * 1.0 / MAX_NUM_THREADS);
    // dim3 unrolling_kernel_grid_dim(gridDimX, Batch, 1);
    // dim3 unrolling_kernel_block_dim(MAX_NUM_THREADS, 1, 1);
    // matrix_unrolling_kernel<<<unrolling_kernel_grid_dim, unrolling_kernel_block_dim>>>(
    //     device_input, unrolled_matrix, Batch, Channel, Height, Width, K
    // );

    int threads = 256;
    size_t num_spatial = (size_t)Height_out * Width_out;
    dim3 block(threads, 1, 1);
    dim3 grid((num_spatial + threads - 1) / threads,
            Channel,
            Batch);
    matrix_unrolling_kernel<<<grid, block>>>(device_input, unrolled_matrix, Batch, Channel, Height, Width, K);

    // Matrix multiplication and permutation. Do not modify.
    // Multiply the mask with the unrolled matrix
    dim3 matmul_grid_dim((Width_unrolled - 1) / MATMUL_TILE_WIDTH + 1,
                         (Map_out - 1) / MATMUL_TILE_WIDTH + 1, 1);
    dim3 matmul_block_dim(MATMUL_TILE_WIDTH, MATMUL_TILE_WIDTH, 1);
    matrixMultiplyShared<<<matmul_grid_dim, matmul_block_dim>>>(
        device_mask, unrolled_matrix, matmul_output, Map_out, Height_unrolled,
        Height_unrolled, Width_unrolled, Map_out, Width_unrolled
    );

    // Permute the result of matrix multiplication
    const size_t out_image_size = Height_out * Width_out;
    dim3 permute_kernel_grid_dim((out_image_size - 1) / PERMUTE_BLOCK_SIZE + 1, Batch, 1);
    matrix_permute_kernel<<<permute_kernel_grid_dim, PERMUTE_BLOCK_SIZE>>>(
        matmul_output, device_output, Map_out, Batch, out_image_size
    );

    cudaFree(matmul_output);
    cudaFree(unrolled_matrix);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host

    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;
    const size_t num_output = (size_t)Batch * Map_out * Height_out * Width_out;
    const size_t bytes_output = num_output * sizeof(float);

    cudaMemcpy(host_output, device_output, bytes_output, cudaMemcpyDeviceToHost);

    // TODO: Free device memory

    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "CUDA error (epilog): " << cudaGetErrorString(error) << std::endl;
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