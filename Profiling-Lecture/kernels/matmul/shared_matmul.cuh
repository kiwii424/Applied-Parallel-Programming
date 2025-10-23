#pragma once

#include "../kernels.hpp"
#include "matmul.hpp"

template<typename ElementType>
struct SharedMatmulKernel {
    using InitParam = GEMMInitParam;
    using InferParam = GEMMInferParam<ElementType>;

    constexpr static unsigned kTileWidth = 16;

    void initialize(InitParam) {}
    void deinitialize() {}

    dim3 getGridSize(InitParam initParam) {
        return dim3((initParam.N + kTileWidth - 1) / kTileWidth, 
                    (initParam.M + kTileWidth - 1) / kTileWidth);
    }

    dim3 getBlockSize(InitParam) {
        return dim3(kTileWidth, kTileWidth);
    }

    __device__ static void kernel(InitParam initParam, InferParam inferParam) {
        __shared__ ElementType subtileA[kTileWidth][kTileWidth];
        __shared__ ElementType subtileB[kTileWidth][kTileWidth];

        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x; 
        ElementType final_val = 0;

        for (unsigned i = 0; i < (initParam.K - 1) / kTileWidth + 1; i++) {
            if (i * kTileWidth + threadIdx.x < initParam.K && row < initParam.M) {
                subtileA[threadIdx.y][threadIdx.x] = 
                    inferParam.matrixA[row * initParam.K + i * kTileWidth + threadIdx.x];
            } else {
                subtileA[threadIdx.y][threadIdx.x] = 0;
            }
            if (i * kTileWidth + threadIdx.y < initParam.K && col < initParam.N) {
                subtileB[threadIdx.y][threadIdx.x] = 
                    inferParam.matrixB[(i * kTileWidth + threadIdx.y) * initParam.N + col];
            } else {
                subtileB[threadIdx.y][threadIdx.x] = 0;
            }

            __syncthreads();

            for (int k = 0; k < kTileWidth; k++) {
                final_val += subtileA[threadIdx.y][k] * subtileB[k][threadIdx.x];
            }

            __syncthreads();
        }

        if (col < initParam.N && row < initParam.M) {
            inferParam.matrixC[row * initParam.N + col] = final_val;
        }
    }
};

template<typename T>
using SharedGemm = KernelFunctor<SharedMatmulKernel<T>>;
