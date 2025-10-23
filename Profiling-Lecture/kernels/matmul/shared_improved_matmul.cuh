#pragma once

#include "../kernels.hpp"
#include "matmul.hpp"
#include "../align_buffer.hpp"

template<typename ElementType>
struct SharedImprovedMatmulKernel {
    using InitParam = GEMMInitParam;
    using InferParam = GEMMInferParam<ElementType>;

    constexpr static unsigned kTileWidth = 16;
    constexpr static unsigned kVectorSize = sizeof(uint4) / sizeof(ElementType);

    void initialize(InitParam) {}
    void deinitialize() {}

    dim3 getGridSize(InitParam initParam) {
        return dim3((initParam.N + kTileWidth - 1) / kTileWidth, 
                    (initParam.M + kTileWidth - 1) / kTileWidth);
    }

    dim3 getBlockSize(InitParam) {
        return dim3(kTileWidth, kTileWidth);
    }

    struct VectorType {
       ElementType vec[kVectorSize];
    };

    struct SharedStorage {
        AlignedSharedMemBuffer<ElementType, kTileWidth * kTileWidth> subtileA;
        AlignedSharedMemBuffer<ElementType, kTileWidth * kTileWidth> subtileB;
    };

    __device__ static void kernel(InitParam initParam, InferParam inferParam, SharedStorage &shared_mem) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x; 
        ElementType final_val = 0;

        for (unsigned i = 0; i < (initParam.K - 1) / kTileWidth + 1; ++i) {

            if (i * kTileWidth + threadIdx.x < initParam.K && row < initParam.M) {
                shared_mem.subtileA[threadIdx.y * kTileWidth + threadIdx.x] = 
                    inferParam.matrixA[row * initParam.K + i * kTileWidth + threadIdx.x];
            } else {
                shared_mem.subtileA[threadIdx.y * kTileWidth + threadIdx.x] = 0;
            }

            if (i * kTileWidth + threadIdx.y < initParam.K && col < initParam.N) {
                shared_mem.subtileB[threadIdx.y * kTileWidth + threadIdx.x] = 
                    inferParam.matrixB[(i * kTileWidth + threadIdx.y) * initParam.N + col];
            } else {
                shared_mem.subtileB[threadIdx.y * kTileWidth + threadIdx.x] = 0;
            }

            __syncthreads();

            // here, in order to decrease MIO throttle, we will make sure to always make
            // shared memory access request with 128 bytes, the maximum a request can be,
            // to reduce the total amount of requests (4 times in case of int/float)
            for (unsigned k = 0; k < kTileWidth; k += kVectorSize) {

                // we can only do so on subtileA since B is loaded column-wise
                VectorType tileA = shared_mem.subtileA.template access_vector<VectorType>((threadIdx.y * kTileWidth + k) / kVectorSize);
                for (unsigned iter = 0; iter < kVectorSize; iter++) {
                    final_val += tileA.vec[iter] * shared_mem.subtileB[(k + iter) * kTileWidth + threadIdx.x];
                }
            }
            __syncthreads();
        }

        if (col < initParam.N && row < initParam.M) {
            inferParam.matrixC[row * initParam.N + col] = final_val;
        }
    }
};

template<typename T>
using SharedImprovedGemm = KernelFunctor<SharedImprovedMatmulKernel<T>>;
