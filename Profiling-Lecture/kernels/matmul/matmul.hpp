#pragma once

// represent the following matrix multiplication:
//      matrixA (M,K) @ matrixB (K,N) = matrixC(M,N)
struct GEMMInitParam {
    size_t M = 0;
    size_t N = 0;
    size_t K = 0;
};

template<typename T>
struct GEMMInferParam {
    const T *matrixA = nullptr;
    const T *matrixB = nullptr;
    T *matrixC = nullptr;
};
