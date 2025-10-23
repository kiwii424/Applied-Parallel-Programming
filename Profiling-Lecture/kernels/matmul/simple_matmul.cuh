#pragma once

#include "../kernels.hpp"
#include "matmul.hpp"

template<typename ElementType>
struct SimpleMatmulKernel {
    using InitParam = GEMMInitParam;
    using InferParam = GEMMInferParam<ElementType>;

    constexpr static unsigned kNumThreads = 256;

    void initialize(InitParam) {}
    void deinitialize() {}

    dim3 getGridSize(InitParam initParam) {
        return dim3((initParam.M * initParam.N + kNumThreads - 1) / kNumThreads);
    }

    dim3 getBlockSize(InitParam) {
        return dim3(kNumThreads);
    }

    __device__ static void kernel(InitParam initParam, InferParam inferParam) {
        unsigned targetIndex = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned targetRowIndex = targetIndex / initParam.N;
        unsigned targetColumnIndex = targetIndex % initParam.N;

        if (targetRowIndex < initParam.M) {
            ElementType sum = 0;
            for (unsigned i = 0; i < initParam.K; i++) {
                sum += inferParam.matrixA[targetRowIndex * initParam.K + i] * inferParam.matrixB[targetColumnIndex + i * initParam.N];
            }
            inferParam.matrixC[targetRowIndex * initParam.N + targetColumnIndex] = sum;
        }
    }
};

template<typename T>
using SimpleGemm = KernelFunctor<SimpleMatmulKernel<T>>;
