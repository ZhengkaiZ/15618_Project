#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "CycleTimer.h"
#include "lstm.h"

void
allocateModel(Model* model) {
    cudaMalloc((void **) &model->W_f, Z * H * sizeof(float));
    cudaMalloc((void **) &model->W_i, Z * H * sizeof(float));
    cudaMalloc((void **) &model->W_c, Z * H * sizeof(float));
    cudaMalloc((void **) &model->W_o, Z * H * sizeof(float));
    cudaMalloc((void **) &model->W_y, H * D * sizeof(float));

    cudaMalloc((void **) &model->b_f, H * sizeof(float));
    cudaMalloc((void **) &model->b_i, H * sizeof(float));
    cudaMalloc((void **) &model->b_c, H * sizeof(float));
    cudaMalloc((void **) &model->b_o, H * sizeof(float));
    cudaMalloc((void **) &model->b_y, D * sizeof(float));
}

void
freeModel(Model *model) {
    cudaFree(model->W_f);
    cudaFree(model->W_i);
    cudaFree(model->W_c);
    cudaFree(model->W_o);
    cudaFree(model->W_y);

    cudaFree(model->b_f);
    cudaFree(model->b_i);
    cudaFree(model->b_c);
    cudaFree(model->b_o);
    cudaFree(model->b_y);
}

void
allocateState(State* state) {
    cudaMalloc((void **) &state->h, H * sizeof(float));
    cudaMalloc((void **) &state->c, H * sizeof(float));
}

void
freeState(State* state) {
    cudaFree(state->h);
    cudaFree(state->c);
}

void
allocateHiddenState(HiddenState* state) {
    cudaMalloc((void **) &state->h_f, H * sizeof(float));
    cudaMalloc((void **) &state->h_i, H * sizeof(float));
    cudaMalloc((void **) &state->h_c, H * sizeof(float));
    cudaMalloc((void **) &state->h_o, H * sizeof(float));

    cudaMalloc((void **) &state->X, Z * sizeof(float));
}

void
freeHiddenState(HiddenState* state) {
    cudaFree(state->h_f);
    cudaFree(state->h_i);
    cudaFree(state->h_c);
    cudaFree(state->h_o);

    cudaFree(state->X);
}

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
matrix_multi(float *x, int x_w, int x_h, bool x_trans, float *y, int y_w, int y_h, bool y_trans, float *result) {
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
        result[index_z] += x[index_x] * y[index_y];
    }
}

__global__ void
matrix_multi_forward(float *x, int x_w, int x_h, float *y, int y_w, int y_h, float *result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= y_h)
        return;

    int k;
    result[index] = 0;

    for (k = 0; k < x_h; k++) {
        int index_x = k;
        int index_y = k * y_w + index;
        result[index] += x[index_x] * y[index_y];
    }
}

// Vector Math functions

__global__ void
exp_vector(float *input, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
        return;

    input[index] = exp(input[index]);
}

__global__ void
sigmoid(float *input, float *output, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
        return;

    output[index] = 1 / (1 + exp(-input[index]));
}

__global__ void
dsigmoid(float *input, float *output, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
        return;

    output[index] = input[index] * (1 - input[index]);
}

__global__ void
tanh(float *input, float *output, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
        return;

    output[index] = tanhf(input[index]);
}

__global__ void
dtanh(float *input, float *output, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
        return;

    output[index] = 1 - input[index] * input[index];
}

// Point-wise Math Functions

__global__ void
pointwise_add(float *nums1, float *nums2, float *output, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
        return;

    output[index] = nums1[index] + nums2[index];
}

__global__ void
pointwise_multi(float *nums1, float *nums2, float *output, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
        return;

    output[index] = nums1[index] * nums2[index];
}

__global__ void
devide(float *nums1, float nums2, float *output, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
        return;

    output[index] = nums1[index] / nums2;
}

__global__ void
sum(float* num, float* sum, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= 1)
        return;

    *sum = thrust::reduce(thrust::device, num, num + N);
}

// main functions
void
cell_forward(int *input, State *old_state, State *state, HiddenState *h, float *prob) {
    //dim3 blockDim(256, 1);
    //dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    dim3 lineDim(256, 1);

    dim3 linesH((H - 1) / lineDim.x + 1);
    dim3 linesD((D - 1) / lineDim.x + 1);


    // One-hot encode
    float *X_one_hot = (float *) malloc(sizeof(float) * D);
    for (int i = 0; i < D; i++) {
        X_one_hot[input[i]] = 1.0;
    }

    // Combine input
    cudaMemcpy(h->X, old_state->h, H * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(h->X + H, X_one_hot, D * sizeof(float), cudaMemcpyHostToDevice);

    float *temp, *temp2, *temp3;
    cudaMalloc((void **) &temp, H * sizeof(float));
    cudaMalloc((void **) &temp2, H * sizeof(float));
    cudaMalloc((void **) &temp3, D * sizeof(float));

    // Forget Gate
    // hf = sigmoid(X @ Wf + bf)
    matrix_multi_forward <<< linesH, lineDim >>> (h->X, 1, Z, model.W_f, Z, H, temp);
    cudaThreadSynchronize();
    pointwise_add <<< linesH, lineDim >>> (temp, model.b_f, temp, H);
    cudaThreadSynchronize();
    sigmoid <<< linesH, lineDim >>> (temp, h->h_f, H);
    cudaThreadSynchronize();

    // Input Gate
    // hi = sigmoid(X @ Wi + bi)
    matrix_multi_forward <<< linesH, lineDim >>> (h->X, 1, Z, model.W_i, Z, H, temp);
    cudaThreadSynchronize();
    pointwise_add <<< linesH, lineDim >>> (temp, model.b_i, temp, H);
    cudaThreadSynchronize();
    sigmoid <<< linesH, lineDim >>> (temp, h->h_i, H);
    cudaThreadSynchronize();

    // Detecting input pattern
    // hc = tanh(X @ Wc + bc)
    matrix_multi_forward <<< linesH, lineDim >>> (h->X, 1, Z, model.W_c, Z, H, temp);
    cudaThreadSynchronize();
    pointwise_add <<< linesH, lineDim >>> (temp, model.b_c, temp, H);
    cudaThreadSynchronize();
    tanh <<< linesH, lineDim >>> (temp, h->h_c, H);
    cudaThreadSynchronize();

    // Output Gate
    // ho = sigmoid(X @ Wo + bo)
    matrix_multi_forward <<< linesH, lineDim >>> (h->X, 1, Z, model.W_o, Z, H, temp);
    cudaThreadSynchronize();
    pointwise_add <<< linesH, lineDim >>> (temp, model.b_o, temp, H);
    cudaThreadSynchronize();
    tanh <<< linesH, lineDim >>> (temp, h->h_o, H);
    cudaThreadSynchronize();

    // c = hf * c_old + hi * hc
    // h = ho * tanh(c)
    pointwise_multi <<< linesH, lineDim >>> (h->h_f, old_state->c, temp, H);
    pointwise_multi <<< linesH, lineDim >>> (h->h_i, h->h_c, temp2, H);
    cudaThreadSynchronize();
    pointwise_add <<< linesH, lineDim >>> (temp, temp2, state->c, H);
    cudaThreadSynchronize();
    tanh <<< linesH, lineDim >>> (state->c, temp, H);
    cudaThreadSynchronize();
    pointwise_multi <<< linesH, lineDim >>> (h->h_o, temp, state->h, H);
    cudaThreadSynchronize();

    // y = h @ Wy + by
    matrix_multi_forward <<< linesD, lineDim >>> (state->h, 1, H, model.W_y, H, D, temp3);
    cudaThreadSynchronize();
    pointwise_add <<< linesD, lineDim >>> (temp3, model.b_y, prob, D);
    cudaThreadSynchronize();

    // prob = softmax(y)
    exp_vector <<< linesD, lineDim >>> (prob, D);
    cudaThreadSynchronize();

    float sum_exp;
    dim3 singleDim(1, 1);
    dim3 single(1);
    sum <<< single, singleDim >>> (prob, &sum_exp, D);
    cudaThreadSynchronize();

    devide <<< linesD, lineDim >>> (prob, sum_exp, prob, D);
    cudaThreadSynchronize();

    cudaFree(temp);
    cudaFree(temp2);
    cudaFree(temp3);
}

void
train() {
    int input[D], i;
    for (i = 0; i < D; i++) {
        input[i] = i;
    }
    State old_state, state;
    HiddenState hiddenState;
    float *prob;

    double startTime = CycleTimer::currentSeconds();

    allocateModel(&model);
    allocateState(&old_state);
    allocateState(&state);
    allocateHiddenState(&hiddenState);

    cudaMalloc((void **) &prob, D * sizeof(float));
    cell_forward(input, &old_state, &state, &hiddenState, prob);

    double endTime = CycleTimer::currentSeconds();

    double overallDuration = endTime - startTime;
    printf("Overall: %.3f ms\t\n", 1000.f * overallDuration);
}

void
printCudaInfo() {

    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
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
