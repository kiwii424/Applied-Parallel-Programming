#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3

//@@ Define constant memory for device kernel here
__constant__ float deviceKernel[MASK_WIDTH * MASK_WIDTH * MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= x_size || y >= y_size || z >= z_size) return;

  int outputIndex = z * y_size * x_size + y * x_size + x;
  output[outputIndex] = 0.0f;

  int halfWidth = MASK_WIDTH / 2;

  for (int i = -halfWidth; i <= halfWidth; ++i) {
    for (int j = -halfWidth; j <= halfWidth; ++j) {
      for (int k = -halfWidth; k <= halfWidth; ++k) {
        int zz = z + i;
        int yy = y + j;
        int xx = x + k;

        if (zz >= 0 && zz < z_size && yy >= 0 && yy < y_size && xx >= 0 && xx < x_size) {
          int inputIndex = zz * y_size * x_size + yy * x_size + xx;
          int kernelIndex = (i + halfWidth) * MASK_WIDTH * MASK_WIDTH + 
                            (j + halfWidth) * MASK_WIDTH + 
                            (k + halfWidth);
          output[outputIndex] += input[inputIndex] * deviceKernel[kernelIndex];
        }
      }
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  //@@ Initial deviceInput and deviceOutput here.
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);


  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  wbCheck(cudaMalloc((void **)&deviceInput, (inputLength - 3) * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, (inputLength - 3) * sizeof(float)));



  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  wbCheck(cudaMemcpy(deviceInput, &hostInput[3], (inputLength - 3) * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpyToSymbol(deviceKernel, hostKernel, MASK_WIDTH * MASK_WIDTH * MASK_WIDTH * sizeof(float)));



  //@@ Initialize grid and block dimensions here
  dim3 DimGrid((x_size + 7) / 8, (y_size + 7) / 8, (z_size + 7) / 8);
  dim3 DimBlock(8, 8, 8);

  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

  cudaDeviceSynchronize();



  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbCheck(cudaMemcpy(&hostOutput[3], deviceOutput, (inputLength - 3) * sizeof(float),
                     cudaMemcpyDeviceToHost));



  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  //@@ Free device memory
  wbCheck(cudaFree(deviceInput));
  wbCheck(cudaFree(deviceOutput));

  // Free host memory
  free(hostInput);
  free(hostOutput);
  free(hostKernel);
  return 0;
}

