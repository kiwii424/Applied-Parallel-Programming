#include <iostream>
#include <mma.h>
#include <cmath>
#include "gpu-new-forward.h"

using namespace nvcuda;

#define TM 16
#define TN 16
#define TK 16
#define BLOCK_T 16

__global__ void conv_wmma_kernel(const float *filter, const float *data, float *result,
                                 int B, int M, int C, int H, int W, int K)
{
    const int Hout = H - K + 1;
    const int Wout = W - K + 1;

    const int UnrollH = C * K * K;
    const int UnrollW = B * Hout * Wout;

    __shared__ __half Asub[TM * TN];
    __shared__ __half Bsub[TM * TN];
    __shared__ float Csub[TM * TN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int global_row = blockIdx.y * BLOCK_T + ty;
    int global_col = blockIdx.x * BLOCK_T + tx;

    wmma::fragment<wmma::matrix_a, TM, TN, TK, __half, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, TM, TN, TK, __half, wmma::row_major> B_frag;
    wmma::fragment<wmma::accumulator, TM, TN, TK, float> Acc;
    wmma::fill_fragment(Acc, 0.0f);

    __half validMask = __float2half(0.0f);

    for (int t = 0; t < (UnrollH + BLOCK_T - 1) / BLOCK_T; t++) {

        int k_idx = t * BLOCK_T + tx;

        int c_idx = k_idx / (K * K);
        int p_idx = (k_idx % (K * K)) / K;
        int q_idx = (k_idx % (K * K)) % K;

        #pragma unroll
        for (int rr = 0; rr < BLOCK_T; rr += 2) {
            int r = global_row + rr;
            validMask = (r < M && k_idx < UnrollH) ? __float2half(1.0f) : __float2half(0.0f);

            Asub[(ty + rr) * BLOCK_T + tx] =
                __float2half(filter[r * (C*K*K) + c_idx * (K*K) + p_idx * K + q_idx]) * validMask;
        }

        int batch_id = global_col / (Hout * Wout);
        int hw = global_col % (Hout * Wout);
        int oh = hw / Wout;
        int ow = hw % Wout;

        #pragma unroll
        for (int rr = 0; rr < BLOCK_T; rr += 2) {

            int row_idx = t * BLOCK_T + ty + rr;
            int c2 = row_idx / (K * K);
            int p2 = (row_idx % (K * K)) / K;
            int q2 = (row_idx % (K * K)) % K;

            int ih = oh + p2;
            int iw = ow + q2;

            validMask = (global_col < UnrollW && row_idx < UnrollH) ? __float2half(1.0f) : __float2half(0.0f);

            Bsub[(ty + rr) * BLOCK_T + tx] =
                __float2half(data[batch_id * (C*H*W) + c2 * (H*W) + ih * W + iw]) * validMask;
        }

        __syncthreads();

        wmma::load_matrix_sync(A_frag, Asub, TM);
        wmma::load_matrix_sync(B_frag, Bsub, TN);
        wmma::mma_sync(Acc, A_frag, B_frag, Acc);

        __syncthreads();
    }

    wmma::store_matrix_sync(Csub, Acc, TM, wmma::mem_row_major);

    __syncthreads();

    #pragma unroll
    for (int rr = 0; rr < BLOCK_T; rr += 2) {
        int r = global_row + rr;
        if (r < M && global_col < UnrollW) {
            int b_id = global_col / (Hout * Wout);
            int hw = global_col % (Hout * Wout);
            int oh = hw / Wout;
            int ow = hw % Wout;

            result[b_id * M * Hout * Wout + r * Hout * Wout + oh * Wout + ow] =
                Csub[(ty + rr) * TM + tx];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *hostOut, const float *hostIn,
                                                    const float *hostF, float **devOut,
                                                    float **devIn, float **devF,
                                                    int B, int M, int C, int H, int W, int K)
{
    size_t in_size = B * C * H * W * sizeof(float);
    size_t f_size = M * C * K * K * sizeof(float);
    size_t out_size = B * M * (H-K+1) * (W-K+1) * sizeof(float);

    cudaMalloc((void**)devIn, in_size);
    cudaMalloc((void**)devF, f_size);
    cudaMalloc((void**)devOut, out_size);

    cudaMemcpy(*devIn, hostIn, in_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*devF, hostF, f_size, cudaMemcpyHostToDevice);
}

__host__ void GPUInterface::conv_forward_gpu(float *devOut, const float *devIn,
                                             const float *devF, int B, int M, int C,
                                             int H, int W, int K)
{
    int Hout = H - K + 1;
    int Wout = W - K + 1;

    int W_unroll = B * Hout * Wout;

    dim3 block(BLOCK_T, 2, 1);
    dim3 grid((W_unroll + BLOCK_T - 1) / BLOCK_T,
              (M + BLOCK_T - 1) / BLOCK_T, 1);

    conv_wmma_kernel<<<grid, block>>>(devF, devIn, devOut,
                                      B, M, C, H, W, K);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *hostOut, float *devOut,
                                                    float *devIn, float *devF,
                                                    int B, int M, int C, int H, int W, int K)
{
    size_t out_size = B * M * (H-K+1) * (W-K+1) * sizeof(float);
    cudaMemcpy(hostOut, devOut, out_size, cudaMemcpyDeviceToHost);

    cudaFree(devOut);
    cudaFree(devIn);
    cudaFree(devF);
}