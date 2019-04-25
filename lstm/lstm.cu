#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"
#include "lstm.h"

// Matrix functions

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

    int k;
    int index_z = index_i * x_w + index_j;
    result[index_z] = 0;

    for (k = 0; k < x_h; k++) {
        int index_x = index(index_i, k, x_w, x_h, x_trans);
        int index_y = index(k, index_j, y_w, y_h, y_trans);
        result[index_z] += x[index_x] * y[index_y]
    }
}

// Vector Math functions

__global__ void
exp_vector(float* input, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
        return;

    input[index] = exp(input[index]));
}

__global__ void
sigmoid(float* input, float* output, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
        return;

    output[index] = 1 / (1 + exp(-input[index]));
}

__global__ void
dsigmoid(float* input, float* output, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
        return;

    output[index] = input[index] * (1 - input[index])
}

__global__ void
tanh(float* input, float* output, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
        return;

    output[index] = tanhf()
}

__global__ void
dtanh(float* input, float* output, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
        return;

    output[index] = 1 - input[index] * input[index];
}

// Point-wise Math Functions

__global__ void
pointwise_add(float* nums1, float* nums2, float* output, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
        return;

    output[index] = nums1[index] + nums2[index];
}

__global__ void
pointwise_multi(float* nums1, float* nums2, float* output, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
        return;

    output[index] = nums1[index] * nums2[index];
}

// main functions
void
cell_forward(int *input, State *old_state, State *state, HiddenState *h, float *prob) {
    // One-hot encode
    float *X_one_hot = (float *) malloc(sizeof(float) * D);
    for (int i = 0; i < Z * H; i++) {
        X_one_hot[input[i]] = 1.0;
    }

    // Combine input
    for (int i = 0; i < H; i++) {
        h->X[i] = old_state->h[i];
    }
    for (int i = H; i < Z; i++) {
        h->X[i] = X_one_hot[i - H];
    }

    float *temp, *temp2, *temp3;
    cudaMalloc((void **) &temp, H * sizeof(float));
    cudaMalloc((void **) &temp2, H * sizeof(float));
    cudaMalloc((void **) &temp3, D * sizeof(float));

    // Forget Gate
    // hf = sigmoid(X @ Wf + bf)
    matrix_multi<<<blocks, threadsPerBlock>>>(h->X, 1, Z, false, model.W_f, Z, H, false, temp);
    cudaThreadSynchronize();
    pointwise_add<<<blocks, threadsPerBlock>>>(temp, model.b_f, temp, H);
    cudaThreadSynchronize();
    sigmoid<<<blocks, threadsPerBlock>>>(temp, h->h_f, H);
    cudaThreadSynchronize();

    // Input Gate
    // hi = sigmoid(X @ Wi + bi)
    matrix_multi<<<blocks, threadsPerBlock>>>(h->X, 1, Z, false, model.W_i, Z, H, false, temp);
    cudaThreadSynchronize();
    pointwise_add<<<blocks, threadsPerBlock>>>(temp, model.b_i, temp, H);
    cudaThreadSynchronize();
    sigmoid<<<blocks, threadsPerBlock>>>(temp, h->h_i, H);
    cudaThreadSynchronize();

    // Detecting input pattern
    // hc = tanh(X @ Wc + bc)
    matrix_multi<<<blocks, threadsPerBlock>>>(h->X, 1, Z, false, model.W_c, Z, H, false, temp);
    cudaThreadSynchronize();
    pointwise_add<<<blocks, threadsPerBlock>>>(temp, model.b_c, temp, H);
    cudaThreadSynchronize();
    tanh<<<blocks, threadsPerBlock>>>(temp, h->h_c, H);
    cudaThreadSynchronize();

    // Output Gate
    // ho = sigmoid(X @ Wo + bo)
    matrix_multi<<<blocks, threadsPerBlock>>>(h->X, 1, Z, false, model.W_o, Z, H, false, temp);
    cudaThreadSynchronize();
    pointwise_add<<<blocks, threadsPerBlock>>>(temp, model.b_o, temp, H);
    cudaThreadSynchronize();
    tanh<<<blocks, threadsPerBlock>>>(temp, h->h_o, H);
    cudaThreadSynchronize();

    // c = hf * c_old + hi * hc
    // h = ho * tanh(c)
    pointwise_multi<<<blocks, threadsPerBlock>>>(h->h_f, old_state->c, temp, H);
    pointwise_multi<<<blocks, threadsPerBlock>>>(h->h_i, h->h_c, temp2, H);
    cudaThreadSynchronize();
    pointwise_add<<<blocks, threadsPerBlock>>>(temp, temp2, state->c, H);
    cudaThreadSynchronize();
    tanh<<<blocks, threadsPerBlock>>>(state->c, temp, H);
    cudaThreadSynchronize();
    pointwise_multi<<<blocks, threadsPerBlock>>>(h->h_o, temp, state->h, H);
    cudaThreadSynchronize();

    // y = h @ Wy + by
    matrix_multi<<<blocks, threadsPerBlock>>>(state->h, 1, H, false, model.W_y, H, D, false, temp3);
    cudaThreadSynchronize();
    pointwise_add<<<blocks, threadsPerBlock>>>(temp3, model.b_y, prob, D);
    cudaThreadSynchronize();

    // prob = softmax(y)
    exp_vector<<<blocks, threadsPerBlock>>>(prob, D);
    cudaThreadSynchronize();
    float sum = sum(prob); // TODO: TBI
    divide(prob, denominator, D);   // TODO: TBI
}

void // TODO: TBC
cell_backward(Model *grad, float *prob, int y_train, State *old_state, State *state, State *new_state,
              HiddenState hiddenState) {
    float *dh_next = new_state->h;
    float *dc_next = new_state->c;

    // Softmax loss gradient
    float *dy = (float *) malloc(sizeof(float) * H);
    memcpy(dy, prob, H);
    dy[y_train] -= 1.0;

    // Hidden to output gradient
    grad->W_y = matrixMultiTrans(state->h, 1, H, true, dy, 1, D, false);
    grad->b_y = dy;
    float *dh = matrixMultiTrans(dy, 1, D, false, model.W_y, H, D, true);

    for (int i = 0; i < H; i++) {
        dh[i] += dh_next[i];
    }

    // Gradient for h_o in
    // h = h_o * tanh(c)
    float *dho = (float *) malloc(H * sizeof(float));
    for (int i = 0; i < H; i++) {
        dho[i] = tanhf(state->c[i]) * dh[i] * dsigmoid(hiddenState.h_o[i]);
    }

    // Gradient for c in
    // h = h_o * tanh(c)
    float *dc = (float *) malloc(H * sizeof(float));
    for (int i = 0; i < H; i++) {
        dc[i] = hiddenState.h_o[i] * dh[i] * dtanh(state->c[i]) + dc_next[i];
    }

    // Gradient for h_f in
    // c = h_f * c_old + h_i * h_c
    float *dhf = (float *) malloc(H * sizeof(float));
    for (int i = 0; i < H; i++) {
        dhf[i] = old_state->c[i] * dc[i] * dsigmoid(hiddenState.h_f[i]);
    }

    // Gradient for h_i in
    // c = h_f * c_old + h_i * h_c
    float *dhi = (float *) malloc(H * sizeof(float));
    for (int i = 0; i < H; i++) {
        dhi[i] = hiddenState.h_c[i] * dc[i] * dsigmoid(hiddenState.h_i[i]);
    }

    // Gradient for h_c in
    // c = h_f * c_old + h_i * h_c
    float *dhc = (float *) malloc(H * sizeof(float));
    for (int i = 0; i < H; i++) {
        dhc[i] = hiddenState.h_i[i] * dc[i] * dtanh(hiddenState.h_c[i]);
    }

    // Gate gradients
    grad->W_f = matrixMultiTrans(hiddenState.X, 1, Z, true, dhf, 1, H, false);
    grad->b_f = dhf;

    grad->W_i = matrixMultiTrans(hiddenState.X, 1, Z, true, dhi, 1, H, false);
    grad->b_i = dhi;

    grad->W_o = matrixMultiTrans(hiddenState.X, 1, Z, true, dho, 1, H, false);
    grad->b_o = dho;

    grad->W_c = matrixMultiTrans(hiddenState.X, 1, Z, true, dhc, 1, H, false);
    grad->b_c = dhc;

    float *dXf = matrixMultiTrans(dhf, 1, H, false, model.W_f, Z, H, true);
    float *dXi = matrixMultiTrans(dhi, 1, H, false, model.W_i, Z, H, true);
    float *dXo = matrixMultiTrans(dho, 1, H, false, model.W_o, Z, H, true);
    float *dXc = matrixMultiTrans(dhc, 1, H, false, model.W_c, Z, H, true);

    for (int i = 0; i < H; i++) {
        dh_next[i] = dXf[i] + dXc[i] + dXi[i] + dXo[i];
        dc_next[i] = hiddenState.h_f[i] * dc[i];
    }
}


// Sample code
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
