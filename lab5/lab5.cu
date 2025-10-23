// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

__global__ void floatToUchar(float *input, unsigned char *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = (unsigned char)(255.0 * input[idx]);
  }
}

__global__ void rgbToGrayscale(unsigned char *input, unsigned char *gray, int width, int height) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int size = width * height;
  if (idx < size) {
    int r = input[3 * idx];
    int g = input[3 * idx + 1];
    int b = input[3 * idx + 2];
    gray[idx] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
  }
}

__global__ void computeHistogram(unsigned char *gray, unsigned int *hist, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ unsigned int temp[HISTOGRAM_LENGTH];

  if (threadIdx.x < HISTOGRAM_LENGTH) {
    temp[threadIdx.x] = 0;
  }
  __syncthreads();

  if (idx < size) {
    atomicAdd(&(temp[gray[idx]]), 1);
  }
  __syncthreads();

  if (threadIdx.x < HISTOGRAM_LENGTH) {
    atomicAdd(&(hist[threadIdx.x]), temp[threadIdx.x]);
  }
}

__global__ void applyEqualization(unsigned char *input, float *cdf, float cdfMin, unsigned char *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float val = 255.0f * (cdf[input[idx]] - cdfMin) / (1.0f - cdfMin);
    val = min(max(val, 0.0f), 255.0f);
    output[idx] = (unsigned char)val;
  }
}

__global__ void ucharToFloat(unsigned char *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input[idx] / 255.0f;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  //Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  int imageSize = imageWidth * imageHeight * imageChannels;
  int graySize = imageWidth * imageHeight;

  float *deviceInputImage, *deviceOutputImage;
  unsigned char *ucharImage, *grayImage, *equalizedImage;
  unsigned int *histogram;
  float *cdf;

  cudaMalloc((void **)&deviceInputImage, imageSize * sizeof(float));
  cudaMalloc((void **)&ucharImage, imageSize * sizeof(unsigned char));
  cudaMalloc((void **)&grayImage, graySize * sizeof(unsigned char));
  cudaMalloc((void **)&histogram, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void **)&cdf, HISTOGRAM_LENGTH * sizeof(float));
  cudaMalloc((void **)&equalizedImage, imageSize * sizeof(unsigned char));
  cudaMalloc((void **)&deviceOutputImage, imageSize * sizeof(float));

  cudaMemcpy(deviceInputImage, hostInputImageData, imageSize * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(histogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));

  int blockSize = 512;
  int numBlocksPixels = (imageSize + blockSize - 1) / blockSize;
  int numBlocksGray = (graySize + blockSize - 1) / blockSize;

  floatToUchar<<<numBlocksPixels, blockSize>>>(deviceInputImage, ucharImage, imageSize);
  cudaDeviceSynchronize();

  rgbToGrayscale<<<numBlocksGray, blockSize>>>(ucharImage, grayImage, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  computeHistogram<<<numBlocksGray, blockSize>>>(grayImage, histogram, graySize);
  cudaDeviceSynchronize();

  unsigned int hostHistogram[HISTOGRAM_LENGTH];
  float hostCdf[HISTOGRAM_LENGTH];

  cudaMemcpy(hostHistogram, histogram, HISTOGRAM_LENGTH * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  hostCdf[0] = hostHistogram[0] / (float)graySize;
  for (int i = 1; i < HISTOGRAM_LENGTH; ++i) {
    hostCdf[i] = hostCdf[i - 1] + hostHistogram[i] / (float)graySize;
  }

  float cdfMin = 1.0f;
  for (int i = 0; i < HISTOGRAM_LENGTH; ++i) {
    if (hostCdf[i] > 0.0f) {
      cdfMin = hostCdf[i];
      break;
    }
  }

  cudaMemcpy(cdf, hostCdf, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyHostToDevice);

  applyEqualization<<<numBlocksPixels, blockSize>>>(ucharImage, cdf, cdfMin, equalizedImage, imageSize);
  cudaDeviceSynchronize();

  ucharToFloat<<<numBlocksPixels, blockSize>>>(equalizedImage, deviceOutputImage, imageSize);
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, deviceOutputImage, imageSize * sizeof(float), cudaMemcpyDeviceToHost);

  wbSolution(args, outputImage);

  cudaFree(deviceInputImage);
  cudaFree(ucharImage);
  cudaFree(grayImage);
  cudaFree(histogram);
  cudaFree(cdf);
  cudaFree(equalizedImage);
  cudaFree(deviceOutputImage);

  return 0;
}

