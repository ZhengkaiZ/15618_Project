#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <pthread.h>

#include "CycleTimer.h"
#include "lstm.h"

#define BLK 16

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

//int BLK = 8;


void showWeights(float *input, int len, char *name) {
    printf("the weights are %s\n", name);
    for (int i = 0; i < len; i++) {
	printf("%f ", input[i]);
    }
    printf("\n");
}



/*************************************************************************
 *
 *  Memory Allocate and Free Functions
 *
 *************************************************************************/


void
allocateModel(Model* model) {
    cudaMalloc((void **) &model->W_f, Z * H * sizeof(float));
    cudaMemset(model->W_f, 0.1, Z * H * sizeof(float));
    
    cudaMalloc((void **) &model->W_i, Z * H * sizeof(float));
    cudaMemset(model->W_i, 0.1, Z * H * sizeof(float));
    
    cudaMalloc((void **) &model->W_c, Z * H * sizeof(float));
    cudaMemset(model->W_c, 0.1, Z * H * sizeof(float));
    
    cudaMalloc((void **) &model->W_o, Z * H * sizeof(float));
    cudaMemset(model->W_o, 0.1, Z * H * sizeof(float));
    
    cudaMalloc((void **) &model->W_y, H * D * sizeof(float));
    cudaMemset(model->W_y, 0.1, D * H * sizeof(float));

    cudaMalloc((void **) &model->b_f, H * sizeof(float));
    cudaMemset(model->b_f, 0.0, H * sizeof(float));

    cudaMalloc((void **) &model->b_i, H * sizeof(float));
    cudaMemset(model->b_i, 0.0, H * sizeof(float));

    cudaMalloc((void **) &model->b_c, H * sizeof(float));
    cudaMemset(model->b_c, 0.0, H * sizeof(float));

    cudaMalloc((void **) &model->b_o, H * sizeof(float));
    cudaMemset(model->b_o, 0.0, H * sizeof(float));

    cudaMalloc((void **) &model->b_y, D * sizeof(float));
    cudaMemset(model->b_y, 0.0, D * sizeof(float));
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
    cudaMemset(state->h, 0.0, H * sizeof(float));
    cudaMalloc((void **) &state->c, H * sizeof(float));
    cudaMemset(state->c, 0.0, H * sizeof(float));
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

int updiv(int n, int d) {
    return (n + d - 1) / d;
}

/*************************************************************************
 *
 *  Matrix Functions
 *
 *************************************************************************/
__device__ float
exp_vector_p(float input) {
    return exp(input);
}

__device__ float
sigmoid_p(float input) {
    return 1 / (1 + exp(-input));
}

__device__ float
dsigmoid_p(float input) {
    return input * (1 - input);
}

__device__ float
tanh_p(float input) {
    return tanhf(input);
}

__device__ float
dtanh_p(float input) {
    return 1 - input * input;
}

__device__ int
RM(int r, int c, int width, int height, bool trans) {
    if (trans) {
	return c * height + r;
    }
    return r * width + c;
}

__global__ void 
cudaBlockKernel(float *matA, int h_A, int w_A, bool x_trans, float *matB, int h_B, int w_B, bool y_trans, float *result) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int hA, wA, hB, wB;

    int bi = threadIdx.y;
    int bj = threadIdx.x;
 
    if (x_trans) {
        hA = w_A;
	wA = h_A;
    } else {
	hA = h_A;
	wA = w_A;
    }
    if (y_trans) {
        hB = w_B;
	wB = h_B;
    } else {
	hB = h_B;
	wB = w_B;
    }

    __shared__ float subA[BLK * BLK];
    __shared__ float subB[BLK * BLK];
    float sum = 0;

    for (int k = 0; k < wA; k += BLK) {
	if (i < hA && k + bj < wA) {
	    subA[RM(bi, bj, BLK, BLK, false)] = matA[RM(i, k + bj, wA, hA, x_trans)];
	} else {
	    subA[RM(bi, bj, BLK, BLK, false)] = 0.0;
	}
	if (j < wB && k + bi < hB) {
	    subB[RM(bi, bj, BLK, BLK, false)] = matB[RM(k + bi, j, wB, hB, y_trans)];
	} else {
	    subB[RM(bi, bj, BLK, BLK, false)] = 0.0;
	}
	__syncthreads();

	for (int bk = 0; bk < BLK; bk++) {
	    sum += subA[RM(bi, bk, BLK, BLK, false)] * subB[RM(bk, bj, BLK, BLK, false)];
	}
	__syncthreads();
    }
    if (i < hA && j < wB) 
    	result[RM(i, j, wB, hA, false)] = sum;
}

__global__ void 
cudaBlockKernel_sigmoid(float *matA, int h_A, int w_A, bool x_trans, float *matB, int h_B, int w_B, bool y_trans, float *matC, float *result) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int hA, wA, hB, wB;

    int bi = threadIdx.y;
    int bj = threadIdx.x;
 
    if (x_trans) {
        hA = w_A;
	wA = h_A;
    } else {
	hA = h_A;
	wA = w_A;
    }
    if (y_trans) {
        hB = w_B;
	wB = h_B;
    } else {
	hB = h_B;
	wB = w_B;
    }

    __shared__ float subA[BLK * BLK];
    __shared__ float subB[BLK * BLK];
    float sum = 0;

    for (int k = 0; k < wA; k += BLK) {
	if (i < hA && k + bj < wA) {
	    subA[RM(bi, bj, BLK, BLK, false)] = matA[RM(i, k + bj, wA, hA, x_trans)];
	} else {
	    subA[RM(bi, bj, BLK, BLK, false)] = 0.0;
	}
	if (j < wB && k + bi < hB) {
	    subB[RM(bi, bj, BLK, BLK, false)] = matB[RM(k + bi, j, wB, hB, y_trans)];
	} else {
	    subB[RM(bi, bj, BLK, BLK, false)] = 0.0;
	}
	__syncthreads();

	for (int bk = 0; bk < BLK; bk++) {
	    sum += subA[RM(bi, bk, BLK, BLK, false)] * subB[RM(bk, bj, BLK, BLK, false)];
	}
	__syncthreads();
    }
    if (i < hA && j < wB) {
	int pos = RM(i, j, wB, hA, false);
    	result[pos] = sigmoid_p(sum + matC[pos]);
    }
}

__global__ void 
cudaBlockKernel_tanh(float *matA, int h_A, int w_A, bool x_trans, float *matB, int h_B, int w_B, bool y_trans, float *matC, float *result) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int hA, wA, hB, wB;

    int bi = threadIdx.y;
    int bj = threadIdx.x;
 
    if (x_trans) {
        hA = w_A;
	wA = h_A;
    } else {
	hA = h_A;
	wA = w_A;
    }
    if (y_trans) {
        hB = w_B;
	wB = h_B;
    } else {
	hB = h_B;
	wB = w_B;
    }

    __shared__ float subA[BLK * BLK];
    __shared__ float subB[BLK * BLK];
    float sum = 0;

    for (int k = 0; k < wA; k += BLK) {
	if (i < hA && k + bj < wA) {
	    subA[RM(bi, bj, BLK, BLK, false)] = matA[RM(i, k + bj, wA, hA, x_trans)];
	} else {
	    subA[RM(bi, bj, BLK, BLK, false)] = 0.0;
	}
	if (j < wB && k + bi < hB) {
	    subB[RM(bi, bj, BLK, BLK, false)] = matB[RM(k + bi, j, wB, hB, y_trans)];
	} else {
	    subB[RM(bi, bj, BLK, BLK, false)] = 0.0;
	}
	__syncthreads();

	for (int bk = 0; bk < BLK; bk++) {
	    sum += subA[RM(bi, bk, BLK, BLK, false)] * subB[RM(bk, bj, BLK, BLK, false)];
	}
	__syncthreads();
    }
    if (i < hA && j < wB) {
	int pos = RM(i, j, wB, hA, false);
    	result[pos] = tanh_p(sum + matC[pos]);
    }
}

__global__ void 
cudaBlockKernel_add(float *matA, int h_A, int w_A, bool x_trans, float *matB, int h_B, int w_B, bool y_trans, float *matC, float *result) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int hA, wA, hB, wB;

    int bi = threadIdx.y;
    int bj = threadIdx.x;
 
    if (x_trans) {
        hA = w_A;
	wA = h_A;
    } else {
	hA = h_A;
	wA = w_A;
    }
    if (y_trans) {
        hB = w_B;
	wB = h_B;
    } else {
	hB = h_B;
	wB = w_B;
    }

    __shared__ float subA[BLK * BLK];
    __shared__ float subB[BLK * BLK];
    float sum = 0;

    for (int k = 0; k < wA; k += BLK) {
	if (i < hA && k + bj < wA) {
	    subA[RM(bi, bj, BLK, BLK, false)] = matA[RM(i, k + bj, wA, hA, x_trans)];
	} else {
	    subA[RM(bi, bj, BLK, BLK, false)] = 0.0;
	}
	if (j < wB && k + bi < hB) {
	    subB[RM(bi, bj, BLK, BLK, false)] = matB[RM(k + bi, j, wB, hB, y_trans)];
	} else {
	    subB[RM(bi, bj, BLK, BLK, false)] = 0.0;
	}
	__syncthreads();

	for (int bk = 0; bk < BLK; bk++) {
	    sum += subA[RM(bi, bk, BLK, BLK, false)] * subB[RM(bk, bj, BLK, BLK, false)];
	}
	__syncthreads();
    }
    if (i < hA && j < wB) {
	int pos = RM(i, j, wB, hA, false);
    	result[pos] = sum + matC[pos];
    }
}

void
cudaMultMatrixBlocked(float *dmatA, int hA, int wA, bool x_trans, float *dmatB, int hB, int wB, bool y_trans, float *dmatC) {
    dim3 threadsPerBlock(BLK, BLK);
    dim3 blocks(updiv(hA, BLK) + 1, updiv(wB, BLK) + 1);
    cudaBlockKernel<<<blocks, threadsPerBlock>>>(dmatA, hA, wA, x_trans, dmatB, hB, wB, y_trans, dmatC);
}

void 
test_kernel() {
    int hA, wA, hB, wB;
    hA = 11;
    wA = 5;
    hB = 5;
    wB = 1;
    float *A = (float*) malloc(hA * wA * sizeof(float));
    float *B = (float*) malloc(hB * wB * sizeof(float));
    float *C = (float*) malloc(hA * wB * sizeof(float));
    float *matC, *matA, *matB;
    cudaMalloc(&matA, hA * wA * sizeof(float));
    cudaMalloc(&matB, hB * wB * sizeof(float));
    cudaMalloc(&matC, hA * wB * sizeof(float)); 
    for (int i = 0; i < hA * wA; i++) {
    	A[i] = i;
    }
    for (int i = 0; i < hB * wB; i++) {
	B[i] = i;
    }
    cudaMemcpy(matA, A, hA * wA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(matB, B, hB * wB * sizeof(float), cudaMemcpyHostToDevice);
    cudaMultMatrixBlocked(matA, wA, hA, true, matB, wB, hB, true, matC);
    cudaMemcpy(C, matC, hA * wB * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < hA * wB; i++) {
	printf("%f ", C[i]);
    }
    printf("\n");
}

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

__global__ void
change_one_element(float *arr, float value, int index) {
    arr[index] += value;
}

__global__ void
pointwise_tanh_dh_dsigmoid(float *c, float *dh, float *ho, float *result, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index >= N) {
	return;
    }
   
   
    result[index] = tanh_p(c[index]) * dh[index] * dsigmoid_p(ho[index]); 
}

__global__ void
pointwise_ho_dh_dtanh_dc_next(float *ho, float *dh, float *c, float *dc_next, float* result, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N) {
	return;
    }

    result[index] = ho[index] * dh[index] * dtanh_p(c[index]) + dc_next[index];
}

__global__ void
pointwise_cold_dc_dsigmoid(float *nums1, float *nums2, float *nums3, float* output, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N) {
	return;
    }
    
    output[index] = nums1[index] * nums2[index] * dsigmoid_p(nums3[index]);
}

__global__ void
pointwise_hi_dc_dtanh(float *nums1, float *nums2, float *nums3, float* output, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N) {
	return;
    }
    output[index] = nums1[index] * nums2[index] * dtanh_p(nums3[index]);
}

__global__ void
pointwise_ho_tanh(float* nums1, float* nums2, float* output, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N) {
	return;
    }
    output[index] = nums1[index] * tanh_p(nums2[index]);
}

__global__ void
pointwise_mult_add_mult(float* nums1, float* nums2, float* nums3, float *nums4, float *output, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N) {
	return;
    }
    output[index] = nums1[index] * nums2[index] + nums3[index] * nums4[index];
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
increase_float(float* prob, int index) {
    float temp = 1.0;
    cudaMemcpy(&prob[index], &temp, sizeof(float), cudaMemcpyHostToDevice);
}

void
decrease_float(float* device, int index) {
    float temp;
    cudaMemcpy(&temp, &device[index], sizeof(float), cudaMemcpyDeviceToHost);
    temp -= 1.0;
    cudaMemcpy(&device[index], &temp, sizeof(float), cudaMemcpyHostToDevice);
}

/*************************************************************************
 *
 *  Main Processes
 *
 *************************************************************************/

void cuda_show_weights(float *input, int length, char *name) {
    float *print = (float *) malloc(length * sizeof(float));
    cudaMemcpy(print, input, length * sizeof(float), cudaMemcpyDeviceToHost);
    showWeights(print, length, name);
    free(print);
}

void
cell_forward_point_fuse(State *old_state, State *state, HiddenState *h, float *prob) {
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

     // Combine input
    cudaMemcpy(h->X, old_state->h, H * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(h->X + H, prob, D * sizeof(float), cudaMemcpyDeviceToDevice);

    //  ho = sigmoid(X @ Wo + bo);
    cudaBlockKernel_sigmoid <<< lineH, lineDim, 0, stream1 >>> (h->X, 1, Z, false, model.W_o, Z, H, false, model.b_o, h->h_o);
    //  hf = sigmoid(X @ Wf + bf);
    cudaBlockKernel_sigmoid <<< lineH, lineDim, 0, stream0 >>> (h->X, 1, Z, false, model.W_f, Z, H, false, model.b_f, h->h_f);
    //  hi = sigmoid(X @ Wi + bi);
    cudaBlockKernel_sigmoid <<< lineH, lineDim, 0, stream0 >>> (h->X, 1, Z, false, model.W_i, Z, H, false, model.b_i, h->h_i);
    //  hc = tanh(X @ Wc + bc);
    cudaBlockKernel_tanh <<< lineH, lineDim, 0, stream0 >>> (h->X, 1, Z, false, model.W_c, Z, H, false, model.b_c, h->h_c);

    // c = hf * c_old + hi * hc
    cudaStreamSynchronize(stream0);
    pointwise_mult_add_mult <<<lineH, lineDim, 0, stream1 >>> (h->h_f, old_state->c, h->h_i, h->h_c, state->c, H);
    // h = ho * tanh(C)
    cudaStreamSynchronize(stream1);
    pointwise_ho_tanh <<< lineH, lineDim, 0, stream0 >>>(h->h_o, state->c, state->h, H);
    
    //prob = softmax()
    cudaStreamSynchronize(stream0);
    cudaBlockKernel_add <<<lineD, lineDim>>> (state->h, 1, H, false, model.W_y, H, D, false, model.b_y, prob);
    cudaThreadSynchronize();

   // cuda_show_weights(prob, D, "probs");
    exp_vector <<< lineD, lineDim >>> (prob, D);
    cudaThreadSynchronize();
    
    //cuda_show_weights(prob, D, "probs");
    float sum_exp;
    sum <<< single, singleDim >>> (prob, &sum_exp, D);
    cudaThreadSynchronize();

    devide <<< lineD, lineDim >>> (prob, sum_exp, prob, D);
    cudaThreadSynchronize();
}

void
cell_forward(State *old_state, State *state, HiddenState *h, float *prob) {
    // Combine input
    cudaMemcpy(h->X, old_state->h, H * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(h->X + H, prob, D * sizeof(float), cudaMemcpyDeviceToDevice);

    float *temp;
    int len = H > D? H: D;
    cudaMalloc((void **) &temp, len * sizeof(float));
    cudaMemset(temp, 0, len * sizeof(float));
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
    //cuda_show_weights(h->h_o, H, "h_o");
     
    cudaThreadSynchronize();
    pointwise_add <<< lineH, lineDim >>> (h->h_f, model.b_f, h->h_f, H);
    pointwise_add <<< lineH, lineDim >>> (h->h_i, model.b_i, h->h_i, H);
    pointwise_add <<< lineH, lineDim >>> (h->h_c, model.b_c, h->h_c, H);
    pointwise_add <<< lineH, lineDim >>> (h->h_o, model.b_o, h->h_o, H);
    cudaThreadSynchronize();

   // cuda_show_weights(h->h_o, H, "h_o");
     
    sigmoid <<< lineH, lineDim >>> (h->h_f, h->h_f, H);
    sigmoid <<< lineH, lineDim >>> (h->h_i, h->h_i, H);
    tanh <<< lineH, lineDim >>> (h->h_c, h->h_c, H);
    sigmoid <<< lineH, lineDim >>> (h->h_o, h->h_o, H);
    cudaThreadSynchronize();

  //  cuda_show_weights(h->h_o, H, "ho");
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

   // cuda_show_weights(h->h_o, H, "h_o");
     
    // y = h @ Wy + by
    matrix_multi_single <<< lineD, lineDim >>> (state->h, model.W_y, H, D, temp);
    cudaThreadSynchronize();
    pointwise_add <<< lineD, lineDim >>> (temp, model.b_y, prob, D);
    cudaThreadSynchronize();

   // cuda_show_weights(h->h_o, H, "h_o");
     
    float sum_exp;
    // prob = softmax(y)
    exp_vector <<< lineD, lineDim >>> (prob, D);
    cudaThreadSynchronize();

   // cuda_show_weights(h->h_o, H, "h_o");
     
    sum <<< single, singleDim >>> (prob, &sum_exp, D);
       
//    sum_exp = thrust::reduce(thrust::device, prob, prob + D);
    //cudaThreadSynchronize();

    devide <<< lineD, lineDim >>> (prob, sum_exp, prob, D);
    cudaThreadSynchronize();

    //cuda_show_weights(model->W_f, D, "W_y");
     
    cudaFree(temp);
}

void cell_backward_point_fuse(Model *grad, float *dy, State *old_state, State *state, State *new_state, 
			  HiddenState *hiddenState) {
    
    float *dh_next = new_state->h;
    float *dc_next = new_state->c;

    float *dh, *dc, *temp;
    cudaMalloc((void **) &dh, H * sizeof(float));
    cudaMalloc((void **) &dc, H * sizeof(float));
    cudaMalloc((void **) &temp, H * sizeof(float));

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
    //cuda_show_weights(dh, H, "dh");

     
    // Gradient for h_o in
    // h = h_o * tanh(c)
    // dho = tanh(c) * dh * dsigmoid(ho)
    float *dho = grad->b_o;
    tanh <<< lineH, lineDim >>> (state->c, dho, H);
    dsigmoid <<< lineH, lineDim >>> (hiddenState->h_o, temp, H);
    cudaThreadSynchronize();
    pointwise_multi <<< lineH, lineDim >>> (dho, dh, temp, dho, H);
    //cuda_show_weights(dho, H, "dho");

    // Gradient for c in
    // h = h_o * tanh(c)
    // dc = ho * dh * dtanh(c) + dc_next
    dtanh <<< lineH, lineDim >>> (state->c, dc, H);
    cudaThreadSynchronize();
    pointwise_multi <<< lineH, lineDim >>> (hiddenState->h_o, dh, dc, dc, H);
    cudaThreadSynchronize();
    pointwise_add <<< lineH, lineDim >>> (dc_next, dc, dc, H);
    //cuda_show_weights(dc, H, "dc");

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
    //cuda_show_weights(dhf, H, "dhf");
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
    //cuda_show_weights(grad->W_f, H, "w_c");
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
    //cuda_show_weights(dX, Z, "X");
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

typedef struct {
    State *old_state;
    State *state;
    HiddenState *h;
    float *prob;
}forward_arg;

typedef struct {
    Model *grad;
    float *dy;
    State *old_state;
    State *state;
    State *new_state;
    HiddenState *hiddenState;
}backward_arg;


void *forward_thread(void *arg) {
    forward_arg* connfd = (forward_arg *) arg;
    cell_forward_point_fuse(connfd->old_state, connfd->state, connfd->h, connfd->prob);
    return NULL;
}


void *backward_thread(void *arg) {
    backward_arg* connfd = (backward_arg *) arg;
    cell_backward(connfd->grad, connfd->dy, connfd->old_state, connfd->state, connfd->new_state, connfd->hiddenState);
    return NULL;
}

void
train(int *X, int *Y, Model *grad, State **states, HiddenState **hiddenState, State *d_next, float **probs) {
    cudaStream_t stream[2];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);
    int min = TIME < LAYER? TIME: LAYER;
    pthread_t threads[min];


    int t, l, i;
    for (t = 1; t <= TIME; t++) {
        cudaMemsetAsync(probs[t-1], 0, D * sizeof(float), stream[0]);               // Initialize prob to 0
    }

    // Gradient for dh_next and dc_next is zero from the last t
    for (l = 0; l < LAYER; l++) {
        cudaMemsetAsync(d_next[l].h, 0, H * sizeof(float), stream[1]);
        cudaMemsetAsync(d_next[l].c, 0, H * sizeof(float), stream[1]);
    }

    cudaStreamSynchronize(stream[0]);
    for (t = 1; t <= TIME; t++) {
        increase_float(probs[t - 1], X[t-1]);        // get one hot encode
        cudaMemcpyAsync(states[t][0].h, probs[t-1], H * sizeof(float), cudaMemcpyDeviceToDevice, stream[0]);  // Initialize h(0) to input
        cudaMemsetAsync(states[t][0].c, 0, H * sizeof(float), stream[0]);          // Initialize c(0) to 0
    }

    cudaStreamSynchronize(stream[0]);

    // Forward
    for(i = 2; i <= TIME+LAYER; i++) {
        int start = i-LAYER > 1? i-LAYER: 1;
        int end = TIME < i-1? TIME: i-1;
        for (t = start; t <= end; t++) {
            l = i-t;

            forward_arg* arg = (forward_arg*) malloc(sizeof(forward_arg));
            arg->old_state = &states[t-1][l];
            arg->state = &states[t][l];
            arg->h = &hiddenState[t][l];
            arg->prob = probs[t-1];

            if (pthread_create(&threads[t-start], NULL, forward_thread, arg) < 0) {
                fprintf(stderr, "Error creating threadn");
            }
        }

        for (t = 0; t <= end-start; t++) {
            if(pthread_join(threads[t], NULL) < 0) {
                fprintf(stderr, "Error joining threadn");
            }
        }
    }

    // Backward
    for (t = 0; t < TIME; t++) {
        decrease_float(probs[t], Y[t]);
    }
    cudaStreamSynchronize(stream[1]);

    for (i = TIME+LAYER-1; i >0; i--) {
        int start = i-LAYER > 1? i-LAYER: 1;
        int end = TIME < i-1? TIME: i-1;
        for (t = end; t >= start; t--) {
            l = i - t;

            cell_backward(grad, probs[t-1], &states[t-1][l], &states[t][l], &d_next[l], &hiddenState[t][l]);

            backward_arg* arg = (backward_arg*) malloc(sizeof(backward_arg));
            arg->grad = grad;
            arg->dy = probs[t-1];
            arg->old_state = &states[t-1][l];
            arg->state = &states[t][l];
            arg->new_state = &d_next[l];
            arg->hiddenState = &hiddenState[t][l];

            if (pthread_create(&threads[t-start], NULL, backward_thread, arg) < 0) {
                fprintf(stderr, "Error creating threadn");
            }
        }

        for (t = end-start; t >= 0 ; t--) {
            if(pthread_join(threads[t], NULL) < 0) {
                fprintf(stderr, "Error joining threadn");
            }
        }
    }

    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
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
setUp(Model *grad, State **states, HiddenState **hiddenState, State *d_next, float **probs) {
    allocateModel(grad);
    int t, l;
    for(t = 0; t <= TIME; t++) {
        cudaHostAlloc((void**)&states[t], sizeof(State) * (LAYER + 1), cudaHostAllocDefault);
        cudaHostAlloc((void**)&hiddenState[t], sizeof(HiddenState) * (LAYER + 1), cudaHostAllocDefault);
        for (l = 0; l <= LAYER; l++) {
            allocateState(&states[t][l]);
            allocateHiddenState(&hiddenState[t][l]);
        }
    }
    for (t = 0; t < TIME; t++)
        cudaMalloc((void **) &probs[t], D * sizeof(float));
    for (l = 0; l < LAYER; l++)
        allocateState(&d_next[l]);
}

void
clean(Model *grad, State **states, HiddenState **hiddenState, State *d_next, float **probs) {
    int t, l;
    // Clean up
    for (t = 0; t <= TIME; t++) {
        for (l = 0; l <= LAYER; l++) {
            freeState(&states[t][l]);
            freeHiddenState(&hiddenState[t][l]);
        }
        cudaFreeHost(states[t]);
        cudaFreeHost(hiddenState[t]);
    }
    for (l = 0; l < LAYER; l++) {
        freeState(&d_next[l]);
    }
    for(t = 0; t < TIME; t++) {
        cudaFree(probs[t]);
    }
    freeModel(grad);
    free(states);
    free(hiddenState);
    free(probs);
}

void
SGD(int **X, int **Y, float learning_rate, int num_samples) {
    double startTime = CycleTimer::currentSeconds();
    int i, j;
    
    Model tmp_grad;
    State **states = (State **) malloc(sizeof(State*) * (TIME+1));
    HiddenState **hiddenState = (HiddenState **) malloc(sizeof(HiddenState*) * (TIME+1));
    float **probs = (float**)malloc(sizeof(float*) * TIME);
    State d_next[LAYER];

    setUp(&tmp_grad, states, hiddenState, d_next, probs);

    for (i = 0; i < EPOCH; i++) {
        for (j = 0; j < num_samples; j++) {
            train(X[j], Y[j], &tmp_grad, states, hiddenState, d_next, probs);
            updateModel(&tmp_grad, learning_rate);
        }
    }

    clean(&tmp_grad, states, hiddenState, d_next, probs);

    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    printf("Overall: %.3f ms\t\n", 1000.f * overallDuration);
}

void
test() {
    int **input, **output, i;
    int num_samples = 5;
    input = (int**)malloc(sizeof(int*) * num_samples);
    output = (int**)malloc(sizeof(int*) * num_samples);
    for (i = 0; i < num_samples; i++) {
        input[i] = (int *)malloc(sizeof(int) * TIME);
        output[i] = (int *)malloc(sizeof(int) * TIME);
	memset(input[i], 0, TIME * sizeof(float));
	memset(output[i], 0, TIME * sizeof(float));
    }
//

    allocateModel(&model);

    SGD(input, output, 1, num_samples);

    freeModel(&model);

}

void
run_large() {
    int num_samples, time;
    int **mat = build_matrix("../data/train.txt", &num_samples, &time);
    //D = 33278; // update D in header file
    int **input = (int **) malloc(num_samples * sizeof(int *));
    int **output = (int**) malloc(num_samples * sizeof(int *));
    for (int i = 0; i < num_samples; i++) {
	input[i] = mat[i];
	output[i] = mat[i] + 1;	
    }
    time -= 1;
//    TIME = 69; // update time in header file
 
    SGD(input, output, 1, 1);
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

int** build_matrix(char *path, int* x, int* y) {
    int i;
    int j;

    FILE *file;
    file = fopen(path, "r");
    int lenX;
    int lenY;
    char p;
    fscanf(file, "%d", &lenX);
    fscanf(file, "%c", &p);
    fscanf(file, "%d", &lenY);

    if (lenY == 0) {
        lenY = 1;
    }

    int** mat;
    mat = (int**) malloc(lenX * sizeof(int *));
    for(i = 0; i < lenX; i++) {
        mat[i] = (int *) malloc(lenY * sizeof(int));
    }

    for(i = 0; i < lenX; i++) {
        for(j = 0; j < lenY; j++) {
            fscanf(file, "%d", &mat[i][j]);
            fscanf(file, "%c", &p);
           // printf("%d ", mat[i][j]);
            if (p == '\n') {
                break;
            }
        }
    }

    *x = lenX;
    if (lenY == 1) {
        *y = lenY - 1;
    } else {
        *y = lenY;
    }

    fclose(file);
    return mat;
}


