#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "CycleTimer.h"
#include "lstm.h"


dim3 singleDim(1, 1);
dim3 lineDim(256, 1);
dim3 rowDim(1, 256);
dim3 gridsDim(16, 16);

dim3 single(1);
dim3 lineH((H - 1) / lineDim.x + 1);
dim3 lineD((D - 1) / lineDim.x + 1);
dim3 lineZ((Z - 1) / lineDim.x + 1);
dim3 lineZH((Z*H - 1) / lineDim.x + 1);
dim3 lineHD((H*D - 1) / lineDim.x + 1);
dim3 rowD(1, (D - 1) / lineDim.x + 1);
dim3 rowZ(1, (Z - 1) / lineDim.x + 1);
dim3 gridHD((H-1)/gridsDim.x+1, (D-1)/gridsDim.y+1);
dim3 gridZH((Z-1)/gridsDim.x+1, (H-1)/gridsDim.y+1);


/*************************************************************************
 *
 *  Memory Allocate and Free Functions
 *
 *************************************************************************/

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

/*************************************************************************
 *
 *  Matrix Functions
 *
 *************************************************************************/

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


// Matrix multi specifically used when 1st matrix x_w is 1.

__global__ void
matrix_multi_single(float *x, float *y, int y_w, int y_h, float *result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= y_h)
        return;

    int k;
    result[index] = 0;

    for (k = 0; k < y_w; k++) {
        int index_x = k;
        int index_y = k * y_w + index;
        result[index] += x[index_x] * y[index_y];
    }
}

/*************************************************************************
 *
 *  Vector Math Functions
 *
 *************************************************************************/

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

/*************************************************************************
 *
 *  Point-wise Math Functions
 *
 *************************************************************************/

__global__ void
pointwise_add(float *nums1, float *nums2, float *output, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
        return;

    output[index] = nums1[index] + nums2[index];
}

__global__ void
pointwise_add(float *nums1, float *nums2, float *num3, float *num4, float *output, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
        return;

    output[index] = nums1[index] + nums2[index] + num3[index] + num4[index];
}

__global__ void
pointwise_multi(float *nums1, float *nums2, float *output, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
        return;

    output[index] = nums1[index] * nums2[index];
}

__global__ void
pointwise_multi(float *nums1, float *nums2, float *num3, float *output, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
        return;

    output[index] = nums1[index] * nums2[index] * num3[index];
}

__global__ void
pointwise_update(float* m, float* grad, int N, float rate) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
        return;

    m[index] += grad[index] * rate;
}

/*************************************************************************
 *
 *  Other Math Functions
 *
 *************************************************************************/

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

void
increase_float(float* device_int) {
    float temp;
    cudaMemcpy(&temp, device_int, sizeof(float), cudaMemcpyDeviceToHost);
    temp += 1.0;
    cudaMemcpy(device_int, &temp, sizeof(float), cudaMemcpyHostToDevice);
}

void
decrease_float(float* device_int) {
    float temp;
    cudaMemcpy(&temp, device_int, sizeof(float), cudaMemcpyDeviceToHost);
    temp -= 1.0;
    cudaMemcpy(device_int, &temp, sizeof(float), cudaMemcpyHostToDevice);
}

/*************************************************************************
 *
 *  Main Processes
 *
 *************************************************************************/

void
cell_forward(State *old_state, State *state, HiddenState *h, float *prob) {
    // Combine input
    cudaMemcpy(h->X, old_state->h, H * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(h->X + H, prob, D * sizeof(float), cudaMemcpyDeviceToDevice);

    float *temp;
    int len = H > D? H: D;
    cudaMalloc((void **) &temp, len * sizeof(float));

    // Forget Gate
    // hf = sigmoid(X @ Wf + bf)
    matrix_multi_single <<< lineH, lineDim >>> (h->X, model.W_f, Z, H, h->h_f);

    // Input Gate
    // hi = sigmoid(X @ Wi + bi)
    matrix_multi_single <<< lineH, lineDim >>> (h->X, model.W_i, Z, H, h->h_i);

    // Detecting input pattern
    // hc = tanh(X @ Wc + bc)
    matrix_multi_single <<< lineH, lineDim >>> (h->X, model.W_c, Z, H, h->h_c);

    // Output Gate
    // ho = sigmoid(X @ Wo + bo)
    matrix_multi_single <<< lineH, lineDim >>> (h->X, model.W_o, Z, H, h->h_o);

    cudaThreadSynchronize();
    pointwise_add <<< lineH, lineDim >>> (h->h_f, model.b_f, h->h_f, H);
    pointwise_add <<< lineH, lineDim >>> (h->h_i, model.b_i, h->h_i, H);
    pointwise_add <<< lineH, lineDim >>> (h->h_c, model.b_c, h->h_c, H);
    pointwise_add <<< lineH, lineDim >>> (h->h_o, model.b_o, h->h_o, H);
    cudaThreadSynchronize();

    sigmoid <<< lineH, lineDim >>> (h->h_f, h->h_f, H);
    sigmoid <<< lineH, lineDim >>> (h->h_i, h->h_i, H);
    tanh <<< lineH, lineDim >>> (h->h_c, h->h_c, H);
    tanh <<< lineH, lineDim >>> (h->h_o, h->h_o, H);
    cudaThreadSynchronize();

    // c = hf * c_old + hi * hc
    // h = ho * tanh(c)
    pointwise_multi <<< lineH, lineDim >>> (h->h_f, old_state->c, temp, H);
    pointwise_multi <<< lineH, lineDim >>> (h->h_i, h->h_c, state->c, H);
    cudaThreadSynchronize();
    pointwise_add <<< lineH, lineDim >>> (temp, state->c, state->c, H);
    cudaThreadSynchronize();
    tanh <<< lineH, lineDim >>> (state->c, temp, H);
    cudaThreadSynchronize();
    pointwise_multi <<< lineH, lineDim >>> (h->h_o, temp, state->h, H);
    cudaThreadSynchronize();

    // y = h @ Wy + by
    matrix_multi_single <<< lineD, lineDim >>> (state->h, model.W_y, H, D, temp);
    cudaThreadSynchronize();
    pointwise_add <<< lineD, lineDim >>> (temp, model.b_y, prob, D);
    cudaThreadSynchronize();

    float sum_exp;
    // prob = softmax(y)
    exp_vector <<< lineD, lineDim >>> (prob, D);
    cudaThreadSynchronize();
    sum <<< single, singleDim >>> (prob, &sum_exp, D);
    cudaThreadSynchronize();
    devide <<< lineD, lineDim >>> (prob, sum_exp, prob, D);
    cudaThreadSynchronize();

    cudaFree(temp);
}

void
cell_backward(Model *grad, float *dy, State *old_state, State *state, State *new_state,
                          HiddenState *hiddenState) {
    float *dh_next = new_state->h;
    float *dc_next = new_state->c;

    float *dh, *dc, *temp;
    cudaMalloc((void **) &dh, H * sizeof(float));
    cudaMalloc((void **) &dc, H * sizeof(float));
    cudaMalloc((void **) &temp, H * sizeof(float));

    // Hidden to output gradient
    // dh = dy @ Wy.T + dh_next
    matrix_multi <<< rowD, rowDim >>> (dy, 1, D, false, model.W_y, H, D, true, dh);
    // dWy = h.T @ dy
    matrix_multi <<< gridHD, gridsDim >>> (state->h, H, 1, false, dy, 1, D, false, grad->W_y);
    // dby = dy
    cudaMemcpy(dy, grad->b_y, D * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaThreadSynchronize();
    pointwise_add <<< lineH, lineDim >>> (dh, dh_next, dh, H);


    // Gradient for h_o in
    // h = h_o * tanh(c)
    // dho = tanh(c) * dh * dsigmoid(ho)
    float *dho = grad->b_o;
    tanh <<< lineH, lineDim >>> (state->c, dho, H);
    dsigmoid <<< lineH, lineDim >>> (hiddenState->h_o, temp, H);
    cudaThreadSynchronize();
    pointwise_multi <<< lineH, lineDim >>> (dho, dh, temp, dho, H);


    // Gradient for c in
    // h = h_o * tanh(c)
    // dc = ho * dh * dtanh(c) + dc_next
    dtanh <<< lineH, lineDim >>> (state->c, dc, H);
    cudaThreadSynchronize();
    pointwise_multi <<< lineH, lineDim >>> (hiddenState->h_o, dh, dc, dc, H);
    cudaThreadSynchronize();
    pointwise_add <<< lineH, lineDim >>> (dc_next, dc, dc, H);


    // Gradient for h_f in
    // c = h_f * c_old + h_i * h_c
    // dhf = c_old * dc * dsigmoid(hf)
    float *dhf = grad->b_f;
    dsigmoid <<< lineH, lineDim >>> (hiddenState->h_f, dhf, H);

    // Gradient for h_i in
    // c = h_f * c_old + h_i * h_c
    // dhi = hc * dc * dsigmoid(hi)
    float *dhi = grad->b_i;
    dsigmoid <<< lineH, lineDim >>> (hiddenState->h_i, dhi, H);

    // Gradient for h_c in
    // c = h_f * c_old + h_i * h_c
    // dhc = hi * dc * dtanh(hc)
    float *dhc = grad->b_c;
    dtanh <<< lineH, lineDim >>> (hiddenState->h_c, dhc, H);

    cudaThreadSynchronize();
    pointwise_multi <<< lineH, lineDim >>> (old_state->c, dc, dhf, dhf, H);
    pointwise_multi <<< lineH, lineDim >>> (hiddenState->h_c, dc, dhi, dhi, H);
    pointwise_multi <<< lineH, lineDim >>> (hiddenState->h_i, dc, dhc, dhc, H);

    // dc_next = hf * dc
    pointwise_multi <<< lineH, lineDim >>> (hiddenState->h_f, dc, dc_next, H);

    // Gate gradients
    // dWf = X.T @ dhf
    matrix_multi <<< gridZH, gridsDim >>> (hiddenState->X, Z, 1, false, dhf, 1, H, false, grad->W_f);
    // dWi = X.T @ dhi
    matrix_multi <<< gridZH, gridsDim >>> (hiddenState->X, Z, 1, false, dhi, 1, H, false, grad->W_i);
    // dWo = X.T @ dho
    matrix_multi <<< gridZH, gridsDim >>> (hiddenState->X, Z, 1, false, dho, 1, H, false, grad->W_o);
    // dWc = X.T @ dhc
    matrix_multi <<< gridZH, gridsDim >>> (hiddenState->X, Z, 1, false, dhc, 1, H, false, grad->W_c);

    float *dXf, *dXi, *dXo, *dXc;
    cudaMalloc((void **) &dXf, Z * sizeof(float));
    cudaMalloc((void **) &dXi, Z * sizeof(float));
    cudaMalloc((void **) &dXo, Z * sizeof(float));
    cudaMalloc((void **) &dXc, Z * sizeof(float));
    // dXf = dhf @ Wf.T
    matrix_multi <<< rowZ, rowDim >>> (dhf, 1, H, false, model.W_f, Z, H, true, dXf);
    // dXi = dhi @ Wi.T
    matrix_multi <<< rowZ, rowDim >>> (dhi, 1, H, false, model.W_i, Z, H, true, dXi);
    // dXo = dho @ Wo.T
    matrix_multi <<< rowZ, rowDim >>> (dho, 1, H, false, model.W_o, Z, H, true, dXo);
    // dXc = dhc @ Wc.T
    matrix_multi <<< rowZ, rowDim >>> (dhc, 1, H, false, model.W_c, Z, H, true, dXc);


    // dX = dXo + dXc + dXi + dXf
    float *dX, *X;
    cudaMalloc((void **) &dX, Z * sizeof(float));
    cudaMalloc((void **) &X, D * sizeof(float));
    cudaThreadSynchronize();
    pointwise_add <<< lineZ, lineDim >>> (dXf, dXc, dXi, dXo, dX, Z);

    // dh_next = dX[:, :H]
    cudaThreadSynchronize();
    cudaMemcpy(dX, dh_next, H * sizeof(float), cudaMemcpyDeviceToDevice);
    // prob = dX[H:Z]
    cudaMemcpy(dX+H, dy, D * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaThreadSynchronize();

    cudaFree(dXf);
    cudaFree(dXi);
    cudaFree(dXo);
    cudaFree(dXc);
    cudaFree(dX);

    cudaFree(dh);
    cudaFree(dc);
    cudaFree(temp);
}

void
train(int *X, int *Y, Model *grad) {
    State **states = (State **) malloc(sizeof(State*) * (TIME + 1));
    HiddenState **hiddenState = (HiddenState **) malloc(sizeof(HiddenState*) * (TIME + 1));
    float **probs = (float **) malloc(sizeof(float *) * TIME);
    int i, k, t, l;
    for(i = 0; i <= TIME; i++) {
        states[i] = (State *)malloc(sizeof(State) * (LAYER + 1));
        hiddenState[i] = (HiddenState *)malloc(sizeof(HiddenState) * (LAYER + 1));
        cudaMalloc((void **) &probs[i], D * sizeof(float));
    }
    for (i = 0; i <= LAYER; i++) {
        allocateState(&states[0][i]);
        allocateHiddenState(&hiddenState[0][i]);
    }

    // Forward
    for (t = 1; t <= TIME; t++) {
        allocateState(&states[t][0]);
        allocateHiddenState(&hiddenState[t][0]);

        cudaMemset(probs[t-1], 0, D);               // Initialize prob to 0
        increase_float(&probs[t-1][X[t-1]]);        // get one hot encode
        cudaMemcpy(states[t][0].h, probs[t-1], H * sizeof(float), cudaMemcpyDeviceToDevice);  // Initialize h(0) to input
        cudaMemset(states[t][0].c, 0, H);           // Initialize c(0) to 0

        // Hidden Layer operate at time t
        for (l = 1; l <= LAYER; l++) {
            allocateState(&states[t][l]);
            allocateHiddenState(&hiddenState[t][l]);
            cell_forward(&states[t-1][l], &states[t][l], &hiddenState[t][l], probs[t - 1]);
        }
    }


    // Backward
    // Gradient for dh_next and dc_next is zero from the last t
    State d_next[LAYER];
    for (k = 0; k < LAYER; k++) {
        allocateState(&d_next[k]);
        cudaMemset(d_next[k].h, 0, H);
        cudaMemset(d_next[k].c, 0, H);
    }

    for (t = TIME; t >= 1; t--) {
        decrease_float(&probs[t-1][Y[t-1]]);
        for (l = LAYER - 1; l >= 0; l--) {
            cell_backward(grad, probs[t-1], &states[t-1][l], &states[t][l], &d_next[l], &hiddenState[t][l]);
        }
    }

    // Clean up
    for (t = 0; t <= TIME; t++) {
        for (l = 0; l <= LAYER; l++) {
            freeState(&states[t][l]);
            freeHiddenState(&hiddenState[t][l]);
        }
        if (t != TIME) {
            cudaFree(probs[t]);
        }
        free(states[t]);
        free(hiddenState[t]);
    }
    for (l = 0; l < LAYER; l++) {
        freeState(&d_next[l]);
    }
    free(states);
    free(hiddenState);
    free(probs);
}

void
updateModel(Model* grad, float learning_rate) {
    pointwise_update <<< lineZH, lineDim >>> (model.W_f, grad->W_f, Z*H, learning_rate);
    pointwise_update <<< lineZH, lineDim >>> (model.W_i, grad->W_i, Z*H, learning_rate);
    pointwise_update <<< lineZH, lineDim >>> (model.W_c, grad->W_c, Z*H, learning_rate);
    pointwise_update <<< lineZH, lineDim >>> (model.W_o, grad->W_o, Z*H, learning_rate);
    pointwise_update <<< lineHD, lineDim >>> (model.W_y, grad->W_y, H*D, learning_rate);

    pointwise_update <<< lineH, lineDim >>> (model.b_f, grad->b_f, H, learning_rate);
    pointwise_update <<< lineH, lineDim >>> (model.b_i, grad->b_i, H, learning_rate);
    pointwise_update <<< lineH, lineDim >>> (model.b_c, grad->b_c, H, learning_rate);
    pointwise_update <<< lineH, lineDim >>> (model.b_o, grad->b_o, H, learning_rate);
    pointwise_update <<< lineD, lineDim >>> (model.b_y, grad->b_y, D, learning_rate);
    cudaThreadSynchronize();
}

void
SGD(int **X, int **Y, float learning_rate, int num_samples) {
    int i, j;
    Model tmp_grad;
    allocateModel(&tmp_grad);

    for (i = 0; i < EPOCH; i++) {
        for (j = 0; j < num_samples; j++) {
            train(X[j], Y[j], &tmp_grad);
            updateModel(&tmp_grad, learning_rate);
        }
    }

    freeModel(&tmp_grad);
}

void
test() {
    int *input, *output, i;
    input = (int*)malloc(sizeof(int) * TIME);
    output = (int*)malloc(sizeof(int) * TIME);
    for (i = 0; i < TIME; i++) {
        input[i] = i % D;
        output[i] = i % D;
    }

    double startTime = CycleTimer::currentSeconds();
    allocateModel(&model);

    SGD(&input, &output, 1, 1);

    freeModel(&model);
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
