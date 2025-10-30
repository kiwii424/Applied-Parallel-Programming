// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 1024 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)
__global__ void addBlockSums(float *output, float *scannedSums, int len) {
  int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockIdx.x > 0 && global_id < len) {
    output[global_id] += scannedSums[blockIdx.x - 1];
  }
}
__global__ void scan(float *input, float *output,float *blockSums, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float prefixsum[BLOCK_SIZE*2];
  int id = threadIdx.x;
  int global_id = blockIdx.x * blockDim.x + id;
  
  if (global_id < len) {
    prefixsum[id] = input[global_id];
  } else {
    prefixsum[id] = 0;
  }
  __syncthreads();

  // Upsweep (Reduction)
  for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
    int index = (id + 1) * stride * 2 - 1;
    if (index < BLOCK_SIZE*2 && index-stride>=0) {
      prefixsum[index] += prefixsum[index - stride];
    }
    __syncthreads();
  }

  // Store block sum but keep prefixsum intact for output
  if (id == BLOCK_SIZE - 1) {
    blockSums[blockIdx.x] = prefixsum[id];
  }
  __syncthreads();

  // Downsweep Phase
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
    int index = (id + 1) * stride * 2 - 1;
    if (index+stride < 2*BLOCK_SIZE) {
      prefixsum[index + stride] += prefixsum[index];
    }
    __syncthreads();
  }
  if (global_id < len) {
    output[global_id] = prefixsum[id];
  }
}
int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  // The number of input elements in the input is numElements
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));


  // Allocate GPU memory.
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));


  // Clear output memory.
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));

  // Copy input memory to the GPU.
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  dim3 DimBlock(BLOCK_SIZE,1,1);
  dim3 DimGrid(ceil(numElements*1.0/BLOCK_SIZE),1,1);

  float *d_blockSums, *d_scannedSums, *temp;
  cudaMalloc(&d_blockSums, DimGrid.x * sizeof(float));
  cudaMalloc(&d_scannedSums, DimGrid.x * sizeof(float));
  cudaMalloc(&temp, DimGrid.x * sizeof(float));
  scan<<<DimGrid,DimBlock>>>(deviceInput, deviceOutput, d_blockSums, numElements);
  cudaDeviceSynchronize();
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  if (DimGrid.x > 1) {
    scan<<<1, DimGrid.x>>>(d_blockSums, d_scannedSums, temp, DimGrid.x);
    cudaDeviceSynchronize();
    addBlockSums<<<DimGrid, DimBlock>>>(deviceOutput, d_scannedSums, numElements);
  }
  cudaDeviceSynchronize();
  // Copying output memory to the CPU
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));


  //@@  Free GPU Memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(d_blockSums);
  cudaFree(d_scannedSums);
  cudaFree(temp);

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

