#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "matmul.h"

#define PERMUTE_BLOCK_SIZE 256
#define TILE_WIDTH 16
#define NUM_STREAMS 4


static const float* g_pinned_host_input = nullptr;


__global__ void matrix_unrolling_kernel(const float *input, float *output,
                             const int Batch, const int Channel,
                             const int Height, const int Width,
                             const int K) {

    const int Height_out = Height - K + 1;
    const int Width_out  = Width - K + 1;

    const int h_out = blockIdx.y * TILE_WIDTH + threadIdx.y;
    const int w_out = blockIdx.x * TILE_WIDTH + threadIdx.x;

    const int width_unrolled = Height_out * Width_out;
    const int col_unroll = h_out * Width_out + w_out;

    #define in_3d(c, h, w) input[(c) * (Height * Width) + (h) * Width + (w)]

    if (h_out < Height_out && w_out < Width_out) {
        for (int c = 0; c < Channel; ++c) {
            const int base_row = c * K * K;
            for (int p = 0; p < K; ++p) {
                for (int q = 0; q < K; ++q) {
                    const int row_unroll = base_row + p * K + q;
                    const int h_in = h_out + p;
                    const int w_in = w_out + q;

                    output[row_unroll * width_unrolled + col_unroll] =
                        in_3d(c, h_in, w_in);
                }
            }
        }
    }

    #undef in_3d
}


// dont need change
__global__ void matrix_permute_kernel(const float *input, float *output,
                           int Map_out, int Batch, int image_size) {

    size_t b = blockIdx.y;
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < image_size) {
        for (size_t m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                input[m * Batch * image_size + b * image_size + x];
        }
    }
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output,
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
                                           const int K) {
    const int Height_out = Height - K + 1;
    const int Width_out  = Width - K + 1;

    const int image_in_size   = Channel * Height * Width;
    const int image_out_size  = Map_out * Height_out * Width_out;
    const int Height_unrolled = Channel * K * K;
    const int Width_unrolled  = Height_out * Width_out;

    const size_t in_bytes   = (size_t)Batch * image_in_size  * sizeof(float);
    const size_t out_bytes  = (size_t)Batch * image_out_size * sizeof(float);
    const size_t mask_bytes = (size_t)Map_out * Channel * K * K * sizeof(float);

    // Allocate device buffers
    cudaMalloc(device_input_ptr,  in_bytes);
    cudaMalloc(device_output_ptr, out_bytes);
    cudaMalloc(device_mask_ptr,   mask_bytes);

    // Copy mask once
    cudaMemcpy(*device_mask_ptr, host_mask, mask_bytes, cudaMemcpyHostToDevice);

    // Register host_input as pinned memory
    g_pinned_host_input = host_input;
    cudaHostRegister(const_cast<float*>(host_input), in_bytes, cudaHostRegisterDefault);

    // Create streams and per-stream working buffers
    cudaStream_t streams[NUM_STREAMS];
    float *unrolled_buf[NUM_STREAMS];
    float *matmul_buf[NUM_STREAMS];

    for (int s = 0; s < NUM_STREAMS; ++s) {
        cudaStreamCreate(&streams[s]);

        cudaMalloc(&unrolled_buf[s],
                   (size_t)Height_unrolled * Width_unrolled * sizeof(float));

        cudaMalloc(&matmul_buf[s],
                   (size_t)Map_out * Width_unrolled * sizeof(float));
    }

    // Kernel launch parameters
    dim3 unroll_block(TILE_WIDTH, TILE_WIDTH);
    dim3 unroll_grid((Width_out  + TILE_WIDTH - 1) / TILE_WIDTH,
                     (Height_out + TILE_WIDTH - 1) / TILE_WIDTH);

    dim3 matmul_grid((Width_unrolled - 1) / MATMUL_TILE_WIDTH + 1,
                     (Map_out      - 1) / MATMUL_TILE_WIDTH + 1);
    dim3 matmul_block(MATMUL_TILE_WIDTH, MATMUL_TILE_WIDTH);

    const int image_pixels_out = Height_out * Width_out;
    dim3 permute_grid((image_pixels_out - 1) / PERMUTE_BLOCK_SIZE + 1, 1);

    // Streamed processing for each image
    for (int b = 0; b < Batch; ++b) {
        int s = b % NUM_STREAMS;
        cudaStream_t stream = streams[s];

        float *d_in_b  = *device_input_ptr  + (size_t)b * image_in_size;
        float *d_out_b = *device_output_ptr + (size_t)b * image_out_size;
        const float *h_in_b = host_input    + (size_t)b * image_in_size;

        // Async H2D copy
        cudaMemcpyAsync(d_in_b, h_in_b,
                        image_in_size * sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream);

        // Unroll
        matrix_unrolling_kernel<<<unroll_grid, unroll_block, 0, stream>>>(
            d_in_b, unrolled_buf[s],
            1, Channel, Height, Width, K);

        // Matmul
        matrixMultiplyShared<<<matmul_grid, matmul_block, 0, stream>>>(
            *device_mask_ptr, unrolled_buf[s], matmul_buf[s],
            Map_out, Height_unrolled,
            Height_unrolled, Width_unrolled,
            Map_out, Width_unrolled);

        // Permute
        matrix_permute_kernel<<<permute_grid, PERMUTE_BLOCK_SIZE, 0, stream>>>(
            matmul_buf[s], d_out_b,
            Map_out, 1, image_pixels_out);
    }

    cudaDeviceSynchronize();

    // Cleanup stream buffers
    for (int s = 0; s < NUM_STREAMS; ++s) {
        cudaFree(unrolled_buf[s]);
        cudaFree(matmul_buf[s]);
        cudaStreamDestroy(streams[s]);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA Error in req_0 streams: "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output,
                                    const float *device_input,
                                    const float *device_mask,
                                    const int Batch,
                                    const int Map_out,
                                    const int Channel,
                                    const int Height,
                                    const int Width,
                                    const int K) {
    (void)device_output;
    (void)device_input;
    (void)device_mask;
    (void)Batch;
    (void)Map_out;
    (void)Channel;
    (void)Height;
    (void)Width;
    (void)K;
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output,
                                           float *device_output,
                                           float *device_input,
                                           float *device_mask,
                                           const int Batch,
                                           const int Map_out,
                                           const int Channel,
                                           const int Height,
                                           const int Width,
                                           const int K) {

    const int Ho = Height - K + 1;
    const int Wo = Width  - K + 1;

    const size_t out_bytes =
        (size_t)Batch * Map_out * Ho * Wo * sizeof(float);

    cudaMemcpy(host_output, device_output, out_bytes, cudaMemcpyDeviceToHost);

    // Unregister pinned memory
    if (g_pinned_host_input != nullptr) {
        cudaHostUnregister(const_cast<float*>(g_pinned_host_input));
        g_pinned_host_input = nullptr;
    }

    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}

// dont need change
__host__ void GPUInterface::get_device_properties() {
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