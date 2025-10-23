#pragma once

#include <cstdio>
#include <iostream>
#include <type_traits>

template<typename... T>
using void_t = void;

// struct used to detect whether a kernel functor needs shared memory
template <typename T, typename = void>
struct has_shared_storage : std::false_type {};

template <typename T>
struct has_shared_storage<T, void_t<typename T::SharedStorage>>
    : std::conditional_t<std::is_void_v<typename T::SharedStorage>, std::false_type, std::true_type> {};

template <typename Kernel>
__global__ void
cudaKernel(typename Kernel::InitParam initParam, typename Kernel::InferParam inferParam) {

    // used to pass in shared memory during runtime
    if constexpr (has_shared_storage<Kernel>::value) {
        // Dynamic shared memory base pointer
        extern __shared__ uint8_t SharedStorageBase[];

        // Declare pointer to dynamic shared memory.
        typename Kernel::SharedStorage *shared_storage =
            reinterpret_cast<typename Kernel::SharedStorage *>(SharedStorageBase);

        Kernel::kernel(initParam, inferParam, *shared_storage);
    } else {
        Kernel::kernel(initParam, inferParam);

    }       
}

template<typename Kernel>
class KernelFunctor {
public:
    // parameter used to initialize kernel
    using InitParam = typename Kernel::InitParam;

    // parameters to the kernel
    using InferParam = typename Kernel::InferParam;

    KernelFunctor() = default;

    bool initialize(InitParam param) {
        if (!_init) {
            _kernel.initialize(param);
            _initParam = param;

            _init = true;
            return true;
        }
        return false;
    }

    void deinitialize() {
        if (_init) {
            _kernel.deinitialize();
        }
        _init = false;
    }

    ~KernelFunctor() {
        deinitialize();
    }
    
    cudaError_t operator()(InferParam inferParam, size_t warmpupIter = 3, size_t profileIter = 100, size_t flop = 0, cudaStream_t stream = (cudaStream_t)(0)) {
        if (!_init) {
            return cudaErrorIllegalState;
        }
        cudaError_t result;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int smem_size = 0;

        if constexpr (has_shared_storage<Kernel>::value) {
            smem_size = int(sizeof(typename Kernel::SharedStorage));
            std::cout << "attempting to allocate " << smem_size << " bytes of shared memory\n";
            if (smem_size >= (48 << 10)) {
                result = cudaFuncSetAttribute(cudaKernel<Kernel>,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                smem_size);

                if (result != cudaSuccess) {
                    return result;
                }
            }
        }

        dim3 _grid = _kernel.getGridSize(_initParam);
        dim3 _block = _kernel.getBlockSize(_initParam);

        std::printf("launching kernel with (%u, %u, %u) (%u, %u, %u)\n", _grid.x, _grid.y, _grid.z, _block.x, _block.y, _block.z);

        for (size_t i = 0; i < warmpupIter; ++i) {
            cudaKernel<Kernel><<<_grid, _block, smem_size, stream>>>(_initParam, inferParam);
        }

        cudaEventRecord(start, stream);

        for (size_t i = 0; i < profileIter; ++i) {
            cudaKernel<Kernel><<<_grid, _block, smem_size, stream>>>(_initParam, inferParam);
        }

        cudaEventRecord(stop, stream);

        result = cudaStreamSynchronize(stream);
        if (result != cudaSuccess) {
            return result;
        }

        if (profileIter != 0) {
            float gemmTime;
            cudaEventElapsedTime(&gemmTime, start, stop);
            gemmTime /= (float)profileIter;

            std::cout << "Time " << gemmTime << " ms\n";
            if (flop != 0) {
                double kernelGflops = flop / 1e6 / gemmTime;
                std::cout << "GFLOPS " << kernelGflops << std::endl;
            }
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return cudaGetLastError();
    }
private:
    Kernel _kernel;
    InitParam _initParam;
    bool _init = false;
};
