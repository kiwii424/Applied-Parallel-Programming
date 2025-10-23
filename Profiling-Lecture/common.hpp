#pragma once

#include <algorithm>
#include <string>
#include <stdexcept>

#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <type_traits>

#define ENABLE_PRINT 0

template <typename T>
void check(T result, char const* const func, const char* const file, int const line);

template <>
void inline check<cudaError_t>(cudaError_t result, char const* const func, const char* const file, int const line) {
  if (result) {
    throw std::runtime_error(std::string("[ERROR] CUDA runtime error: ") +
                             (cudaGetErrorName(result)) + " " + file + ":" +
                             std::to_string(line) + " \n");
  }
}

template <>
void inline check<cublasStatus_t>(cublasStatus_t result, char const* const func, const char* const file, int const line) {
  if (result) {
    throw std::runtime_error(std::string("[ERROR] cuBLAS runtime error: ") +
                             (cublasGetStatusString(result)) + " " + file + ":" +
                             std::to_string(line) + " \n");
  }
}

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

inline int random_int() { return (std::rand() % 100); }

// adapted from https://stackoverflow.com/questions/4915462/how-should-i-do-floating-point-comparison
template<typename T>
__device__ __forceinline__ bool nearly_equal (
  T a, T b,
  T epsilon = 0, float abs_th = 1e-6)
  // those defaults are arbitrary and could be removed
{
  assert(std::numeric_limits<T>::epsilon() <= epsilon);
  assert(epsilon < 1.f);

  if (a == b) return true;

  auto diff = std::abs(a-b);
  auto norm = std::min((std::abs(a) + std::abs(b)), std::numeric_limits<float>::max());
  // or even faster: std::min(std::abs(a + b), std::numeric_limits<float>::max());
  // keeping this commented out until I update figures below
  return diff < std::max(abs_th, epsilon * norm);
}

// Kernel to compare two vectors element-wise
template<typename T>
__global__ void compareVectors(const T* d_ref, const T* d_actual, int n, int* mismatchFlag) {
    constexpr T epsilon = std::is_floating_point_v<T> ? 1e-4 : 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (!nearly_equal(d_ref[idx], d_actual[idx], epsilon)) {
          #if ENABLE_PRINT
            printf("index %d differ! expect %f but have %f\n", idx, d_ref[idx], d_actual[idx]);
          #endif
            // Mark mismatch
            *mismatchFlag = 1;
        }
    }
}

template<typename T>
bool inline vectorsEqual(const T* d_ref, const T* d_actual, int n) {
    int h_mismatchFlag = 0;
    int* d_mismatchFlag;

    cudaMalloc(&d_mismatchFlag, sizeof(int));
    cudaMemcpy(d_mismatchFlag, &h_mismatchFlag, sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    compareVectors<<<gridSize, blockSize>>>(d_ref, d_actual, n, d_mismatchFlag);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_mismatchFlag, d_mismatchFlag, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_mismatchFlag);

    return (h_mismatchFlag == 0);
}

#undef ENABLE_PRINT 
