#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

__device__ int
index(int i, int j, int width, int height, bool column_base) {
    if (column_base) {
        return i * width + j;
    } else {
        return j * height + i;
    }
}

__global__ void
matrix_multi(float* x, int x_w, int x_h, bool x_trans, float* y, int y_w, int y_h, bool y_trans, float* result) {
    int index_i = blockIdx.x * blockDim.x + threadIdx.x;
    int index_j = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_trans) {
        int temp = x_h;
        x_h = x_w;
        x_w = temp;
    }

    if (y_trans) {
        int temp = y_h;
        y_h = y_w;
        y_w = temp;
    }

    if (index_i >= x_w || index_j >= y_h)
        return;

    int i;
    int index_z = index_i * x_w + index_j;
    result[index_z] = 0;

    for (k = 0; k < x_h; k++) {
        int index_x = index(index_i, k, x_w, x_h, x_trans);
        int index_y = index(k, index_j, y_w, y_h, y_trans);
        result[index_z] += x[index_x] * y[index_y]
    }
}

__global__ void
sigmoid(float* input, int N, float* output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
        return;

    output[index] = 1 / (1 + exp(-input[index]))
}

__global__ void
dsigmoid(float* input, int N, float* output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
        return;

    output[index] = input[index] * (1 - input[index])
}

__global__ void
tanh(float* input, int N, float* output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
        return;

    output[index] = tanhf()
}

__global__ void
dtanh(float* input, int N, float* output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
        return;

    output[index] = 1 - input[index] * input[index];
}

void
saxpyCuda(int N, float alpha, float* xarray, float* yarray, float* resultarray) {

    int totalBytes = sizeof(float) * 3 * N;

    // compute number of blocks and threads per block
    const int threadsPerBlock = 512;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* device_x;
    float* device_y;
    float* device_result;

    //
    // allocate device memory buffers on the GPU using cudaMalloc
    //
    cudaMalloc((void **) &device_x, N * sizeof(float));
    cudaMalloc((void **) &device_y, N * sizeof(float));
    cudaMalloc((void **) &device_result, N * sizeof(float));

    // start timing after allocation of device memory
    double startTime = CycleTimer::currentSeconds();

    //
    // copy input arrays to the GPU using cudaMemcpy
    //
    cudaMemcpy(device_x, xarray, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, yarray, N * sizeof(float), cudaMemcpyHostToDevice);

    // run kernel
    double startTimeKernel = CycleTimer::currentSeconds();
    saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);
    cudaThreadSynchronize();
    double endTimeKernel = CycleTimer::currentSeconds();

    //
    // copy result from GPU using cudaMemcpy
    //
    resultarray = (float *) calloc(N, sizeof(float));
    cudaMemcpy(resultarray, device_result, N * sizeof(float), cudaMemcpyDeviceToHost);

    // end timing after result has been copied back into host memory
    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    double overallDuration = endTime - startTime;
    double overallDurationKernel = endTimeKernel - startTimeKernel;
    printf("Overall: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW(totalBytes, overallDuration));
    printf("Kernel Overall: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDurationKernel, toBW(totalBytes, overallDurationKernel));

    // free memory buffers on the GPU
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_result);

}

void
printCudaInfo() {

    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
