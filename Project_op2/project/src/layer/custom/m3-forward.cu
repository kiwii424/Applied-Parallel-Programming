#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define task "op_2"
#define TILE_WIDTH 16

// ------------------------------//
// Fused conv with loop unroll  //
// ------------------------------//
__global__ void matmul_conv_fused(const float *mask,
                                  const float *input,
                                  float *output,
                                  int Batch, int Map_out, int Channel,
                                  int Height, int Width, int K)
{
    /*
    mask   - convolution kernel [Map_out, Channel, K, K]
    input  - input feature maps [Batch, Channel, Height, Width]
    output - output feature maps [Batch, Map_out, H_out, W_out]
    Batch  - batch size
    Map_out- number of output feature maps
    Channel- number of input feature maps
    Height - input height
    Width  - input width
    K      - kernel size (K x K)
    */

    __shared__ float tile_mask[TILE_WIDTH][TILE_WIDTH];   // tile of A
    __shared__ float tile_input[TILE_WIDTH][TILE_WIDTH];  // tile of B

    const int H_out = Height - K + 1;
    const int W_out = Width  - K + 1;
    const int KK    = K * K;
    const int img_sz = H_out * W_out;

    // A: [Map_out, Channel*K*K]
    // B: [Channel*K*K, Batch*H_out*W_out]
    // C: [Map_out,    Batch*H_out*W_out]
    const int numARows = Map_out;
    const int numACols = Channel * KK;
    const int numBRows = numACols;
    const int numBCols = Batch * img_sz;
    const int numCRows = numARows;
    const int numCCols = numBCols;

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int row = blockIdx.y * TILE_WIDTH + ty;
    const int col = blockIdx.x * TILE_WIDTH + tx;

    #define IN4D(b, c, h, w) \
        input[(b) * (Channel * Height * Width) + (c) * (Height * Width) + (h) * Width + (w)]

    #define MASK4D(m, c, p, q) \
        mask[(m) * (Channel * K * K) + (c) * (K * K) + (p) * K + (q)]

    float acc = 0.0f;

    const int numTiles = (numACols + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int t = 0; t < numTiles; ++t) {
        const int kA = t * TILE_WIDTH + tx;  // column in A
        const int kB = t * TILE_WIDTH + ty;  // row in B

        // Load A tile (mask)
        if (row < numARows && kA < numACols) {
            const int c  = kA / KK;
            const int r  = kA - c * KK;   // r = p*K + q
            const int p  = r / K;
            const int q  = r - p * K;
            tile_mask[ty][tx] = MASK4D(row, c, p, q);
        } else {
            tile_mask[ty][tx] = 0.0f;
        }

        // Load B tile (unrolled input)
        if (col < numBCols && kB < numBRows) {
            const int b  = col / img_sz;
            const int hw = col - b * img_sz;
            const int ho = hw / W_out;
            const int wo = hw - ho * W_out;

            const int c2 = kB / KK;
            const int r2 = kB - c2 * KK;
            const int p2 = r2 / K;
            const int q2 = r2 - p2 * K;

            const int hi = ho + p2;
            const int wi = wo + q2;

            tile_input[ty][tx] = IN4D(b, c2, hi, wi);
        } else {
            tile_input[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Inner product with explicit loop unrolling (factor 4)
        if (row < numCRows && col < numCCols) {
            // TILE_WIDTH = 16, so i = 0,4,8,12 are all valid
            #pragma unroll
            for (int i = 0; i < TILE_WIDTH; i += 4) {
                acc += tile_mask[ty][i    ] * tile_input[i    ][tx];
                acc += tile_mask[ty][i + 1] * tile_input[i + 1][tx];
                acc += tile_mask[ty][i + 2] * tile_input[i + 2][tx];
                acc += tile_mask[ty][i + 3] * tile_input[i + 3][tx];
            }
        }

        __syncthreads();
    }

    // Write back to 4D output layout
    if (row < numCRows && col < numCCols) {
        const int b  = col / img_sz;
        const int hw = col - b * img_sz;
        const int ho = hw / W_out;
        const int wo = hw - ho * W_out;

        output[((b * Map_out + row) * H_out + ho) * W_out + wo] = acc;
    }

    #undef IN4D
    #undef MASK4D
}

// ------------------------------//
// Prolog: allocate & memcpy     //
// ------------------------------//
__host__
void GPUInterface::conv_forward_gpu_prolog(const float *host_output,
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
    const int H_out = Height - K + 1;
    const int W_out = Width  - K + 1;

    const size_t in_bytes =
        static_cast<size_t>(Batch) * Channel * Height * Width * sizeof(float);
    const size_t mask_bytes =
        static_cast<size_t>(Map_out) * Channel * K * K * sizeof(float);
    const size_t out_bytes =
        static_cast<size_t>(Batch) * Map_out * H_out * W_out * sizeof(float);

    cudaMalloc(device_input_ptr,  in_bytes);
    cudaMalloc(device_mask_ptr,   mask_bytes);
    cudaMalloc(device_output_ptr, out_bytes);

    cudaMemcpy(*device_input_ptr, host_input, in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr,  host_mask,  mask_bytes, cudaMemcpyHostToDevice);

    std::cout << "Task: " << task << std::endl;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error (prolog): "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// ------------------------------//
// Main convolution entry (GPU)  //
// ------------------------------//
__host__
void GPUInterface::conv_forward_gpu(float *device_output,
                                    const float *device_input,
                                    const float *device_mask,
                                    const int Batch,
                                    const int Map_out,
                                    const int Channel,
                                    const int Height,
                                    const int Width,
                                    const int K)
{
    const int H_out = Height - K + 1;
    const int W_out = Width  - K + 1;
    const int W_unroll = Batch * H_out * W_out;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim((W_unroll + TILE_WIDTH - 1) / TILE_WIDTH,
                 (Map_out  + TILE_WIDTH - 1) / TILE_WIDTH,
                 1);

    matmul_conv_fused<<<gridDim, blockDim>>>(
        device_mask, device_input, device_output,
        Batch, Map_out, Channel, Height, Width, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error (op_2): "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// ------------------------------//
// Epilog: copy back & free      //
// ------------------------------//
__host__
void GPUInterface::conv_forward_gpu_epilog(float *host_output,
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
    const size_t out_bytes =
        static_cast<size_t>(Batch) * Map_out * H_out * W_out * sizeof(float);

    cudaMemcpy(host_output, device_output, out_bytes, cudaMemcpyDeviceToHost);

    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}

// ------------------------------//
// Device properties (unchanged) //
// ------------------------------//
__host__
void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
        std::cout << "Computational capabilities: "
                  << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
        std::cout << "Max Shared memory size per block: "
                  << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0]
                  << " x, " << deviceProp.maxThreadsDim[1]
                  << " y, " << deviceProp.maxThreadsDim[2] << " z" << std::endl;
        std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0]
                  << " x, " << deviceProp.maxGridSize[1]
                  << " y, " << deviceProp.maxGridSize[2] << " z" << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    }
}