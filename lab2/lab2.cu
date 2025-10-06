#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns)
{
  //@@ Implement matrix multiplication kernel here
  int idx_row = blockIdx.y * blockDim.y + threadIdx.y;
  int idx_col = blockIdx.x * blockDim.x + threadIdx.x;

  C[idx_row * numCColumns +idx_col] = 0;
  for(int i = 0; i < numAColumns; i++){
    if(idx_row < numCRows && idx_col < numCColumns){
      C[idx_row * numCColumns + idx_col] += A[idx_row * numAColumns + i] * B[i * numBColumns + idx_col];
    }
  }
}


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  //@@ Importing data and creating memory on host
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;


  //@@ Allocate the hostC matrix
  hostC = (float *) malloc(numCRows * numCColumns * sizeof(float));


  //@@ Allocate GPU memory here
  float *DeviceA, *DeviceB, *DeviceC;
  wbCheck(cudaMalloc((void **) &DeviceA, numAColumns * numARows * sizeof(float)));
  wbCheck(cudaMalloc((void **) &DeviceB, numBColumns * numBRows * sizeof(float)));
  wbCheck(cudaMalloc((void **) &DeviceC, numCColumns * numCRows * sizeof(float)));
  

  //@@ Copy memory to the GPU here
  wbCheck(cudaMemcpy(DeviceA, hostA, numAColumns * numARows * sizeof(float), cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(DeviceB, hostB, numBColumns * numBRows * sizeof(float), cudaMemcpyHostToDevice));


  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid( ceil(numCColumns / 16.0), ceil(numCRows / 16.0), 1);
  dim3 DimBlock( 16, 16, 1);


  //@@ Launch the GPU Kernel here
  matrixMultiply<<<DimGrid, DimBlock>>>(DeviceA, DeviceB, DeviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  cudaDeviceSynchronize();
  
  //@@ Copy the GPU memory back to the CPU here
  wbCheck(cudaMemcpy(hostC, DeviceC, numCColumns * numCRows * sizeof(float), cudaMemcpyDeviceToHost));


  //@@ Free the GPU memory here
  wbCheck(cudaFree(DeviceA));
  wbCheck(cudaFree(DeviceB));
  wbCheck(cudaFree(DeviceC));

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  //@@Free the hostC matrix
  free(hostC);

  return 0;
}

