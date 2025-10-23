#include "argparse.hpp"
#include "common.hpp"

#include "kernels/matmul/matmul.hpp"
#include "kernels/matmul/simple_matmul.cuh"
#include "kernels/matmul/shared_matmul.cuh"
#include "kernels/matmul/shared_improved_matmul.cuh"

#include <nvToolsExt.h>

int main(int argc, char **argv) {
    argparse::Parser parser;

    // default matrix sizes:
    size_t m = 1600;
    size_t n = 1400;
    size_t k = 1500;

    GEMMInitParam initParam{m, n, k};

    size_t nIters = 20;
    size_t nWarmUp = 3;
    bool isProfile = false;
    parser.add_positional(m);
    parser.add_positional(n);
    parser.add_positional(k);
    parser.add_option(nIters,"--iters");
    parser.add_option(nWarmUp,"--warmup");
    parser.add_flag(isProfile, "--profile", "-p");

    if (!parser.parse(argc, argv)) {
        parser.help();
        exit(-1);
    }

    if (isProfile) {
        nWarmUp = 1;
        nIters = 0;
    }

    const size_t flop = int64_t(m) * int64_t(n) * int64_t(k) * 2;

    // initialize host data
    std::cout << "generate data\n";
    nvtxRangePush("generate data");
    float *aHost, *bHost;
    aHost = new float[m * k];
    bHost = new float[k * n];
    std::generate(aHost, aHost + m * k, random_int);
    std::generate(bHost, bHost + k * n, random_int);
    nvtxRangePop();

    // allocate device data
    float *aDev, *bDev;
    float *cMatmulSimple;
    float *cMatmulShared;
    float *cMatmulSharedImproved;

    CHECK_CUDA_ERROR(cudaMalloc(&aDev, m * k * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&bDev, k * n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&cMatmulSimple, m * n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&cMatmulShared, m * n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&cMatmulSharedImproved, m * n * sizeof(float)));

    GEMMInferParam<float> inferParam{aDev, bDev, nullptr};

    // copy data to device
    nvtxRangePush("host-to-device");
    CHECK_CUDA_ERROR(
        cudaMemcpy(aDev, aHost, m * k * sizeof(float), cudaMemcpyDefault));
    CHECK_CUDA_ERROR(
        cudaMemcpy(bDev, bHost, k * n * sizeof(float), cudaMemcpyDefault));
    nvtxRangePop();

    // call all implementations
    cudaProfilerStart();

    // matmul simple
    nvtxRangePush("matmul simple kernel");
    std::cout << "running simple matmul kernel" << std::endl;
    inferParam.matrixC = cMatmulSimple;
    SimpleGemm<float> simpleKernel;
    simpleKernel.initialize(initParam);
    simpleKernel(inferParam, nWarmUp, nIters, flop);
    nvtxRangePop();

    // matmul shared
    nvtxRangePush("matmul shared kernel");
    std::cout << "\n\nrunning shared matmul kernel" << std::endl;
    inferParam.matrixC = cMatmulShared;
    SharedGemm<float> sharedKernel;
    sharedKernel.initialize(initParam);
    sharedKernel(inferParam, nWarmUp, nIters, flop);
    nvtxRangePop();

    // matmul shared improved
    nvtxRangePush("matmul shared improved kernel");
    std::cout << "\n\nrunning shared improved matmul kernel" << std::endl;
    inferParam.matrixC = cMatmulSharedImproved;
    SharedImprovedGemm<float> sharedimprovedKernel;
    sharedimprovedKernel.initialize(initParam);
    sharedimprovedKernel(inferParam, nWarmUp, nIters, flop);
    nvtxRangePop();

    cudaProfilerStop();
    
    CHECK_CUDA_ERROR(cudaFree(aDev));
    CHECK_CUDA_ERROR(cudaFree(bDev));
    CHECK_CUDA_ERROR(cudaFree(cMatmulSimple));
    CHECK_CUDA_ERROR(cudaFree(cMatmulShared));
    CHECK_CUDA_ERROR(cudaFree(cMatmulSharedImproved));

    delete[] aHost;
    delete[] bHost;
    return 0;
}
