#include "util.h"

static void initiateCache() {
    float *cache.X = (float*)malloc(sizeof(float) * (H + D));
    float *cache.h = (float*)malloc(sizeof(float) * H);
    float *cache.c = (float*)malloc(sizeof(float) * H);
    float *cache.c_old = (float*)malloc(sizeof(float) * H);
    float *cache.ho = (float*)malloc(sizeof(float) * H);
    float *cache.hf = (float*)malloc(sizeof(float) * H);
}

static inline float dsigmoid(float input) {
    return input * (1 - input);
}

static inline float dtanh(float input) {
    return 1 - input * input;
}

static inline float* dsigmoid_vector(float *input, int len) {
    int i;
    float *output = (float *) malloc(len * sizeof(float));
    for (i = 0; i < len; i++) {
        output[i] = dsigmoid(input[i]);
    }

    return output[i]
}

static inline float* dtanh_vector(float *input, int len) {
    int i = 0;
    float *output = (float *) malloc(len * sizeof(float));
    for (i = 0;i < len; i++) {
        output[i] = dtanh(input[i]);
    }
    
    return output[i]
}

static inline float* deep_copy(float* input, int input_len) {
    int i = 0;
    float *result = (float *) malloc(input_len * sizeof(float));
    for (i; i < y_len; i++) {
        result[i] = input[i];
    }

    return result;
}

static inline int indexTrans(int i, int j, int height, int width, bool isTrans) {
    if (isTrans) {
        return j * height + i;
    } else {
        return i * width + j;
    }
}

static float* matrixMultiTrans(float *X, int X_h, int X_w, bool X_isTrans, float *Y, int Y_h, int Y_w, bool Y_isTrans) {
    int Xh, Xw, Yh, Yw;
    if (X_isTrans) {
        Xh = X_w;
        Xw = X_h;
    } else {
        Xh = X_h;
        Xw = X_w;
    }
    
    if (Y_isTrans) {
        Yh = Y_w;
        Yw = Y_h;
    } else {
        Yh = Y_h;
        Yw = Y_w;
    }

    float* result = (float*)malloc(sizeof(float) * Xh * Yw);
    for (int i = 0; i < Xh; i++) {
        for (int j = 0; j < Yw; j++) {
            int index_c = indexTrans(i, j, Xh, Yw, false);
            result[index_c] = 0;
            for (int k = 0; k < Xh; k++) {
                int index_a = indexTrans(i, k, Xh, Xw, X_isTrans);
                int index_b = indexTrans(k, j, Yh, Yw, Y_isTrans);
                result[index_c] += X[index_a] * Y[index_b];
            }
        }
    }
    return result;
}


static Model backward(float* prob, int y_train, float* d_next_h, float* d_next_c) {

    float *dh_next = d_next_h;
    float *dc_next = d_next_c;
    float *dy = deep_copy(prob, D);
    dy[y_train] -= 1.0;
    
    grad.W_y = matrixMultiTrans(cache.h, 1, H, true, dy, 1, D, false);
    grad.b_y = dy;
    
    float *dh = matrixMultiTrans(dy, 1, D, false, model.W_y, H, D, true);
    for (int i = 0; i < H; i++) {
        dh[i] += dh_next[i];
    }
    
    float *dho = (float*) malloc(H * sizeof(float));
    for (int i = 0; i < H; i++) {
        dho[i] = tanhf(cache.c[i]) * dh[i] * dsigmoid(cache.ho[i]);
    }
    
    float *dc = (float*) malloc(H * sizeof(float));
    for (int i = 0; i < H; i++) {
        dc[i] = cache.ho[i] * dh[i] * dtanh(cache.c[i]) + dc_next[i];
    }
    
    float *dhf = (float*) malloc(H * sizeof(float));
    for (int i = 0; i < H; i++) {
        dhf[i] = cache.c_old[i] * dc[i] * dsigmoid[cache.hf[i]];
    }
    
    float *dhi = (float*) malloc(H * sizeof(float));
    for (int i = 0; i < H; i++) {
        dhi[i] = cache.hc[i] * dc[i] * dsigmoid(cache.hi[i]);
    }
    
    float *dhc = (float*) malloc(H * sizeof(float));
    for (int i = 0; i < H; i++) {
        dhc[i] = cache.hi[i] * dc[i] * dtanh(cache.hc[i])
    }
    
    grad.W_f = matrixMultiTrans(cache.X, 1, Z, true, dhf, 1, H, false);
    grad.b_f = dhf;
    
    grad.W_i = matrixMultiTrans(cache.X, 1, Z, true, dhi, 1, H, false);
    grad.b_i = dhi;
    
    grad.W_o = matrixMultiTrans(cache.X, 1, Z, true, dho, 1, H, false);
    grad.b_o = dho;
    
    grad.W_c = matrixMultiTrans(cache.X, 1, Z, true, dhc, 1, H, false);
    grad.b_c = dhc;
    
    float *dXf = matrixMultiTrans(dhf, 1, H, false, cache.W_f, Z, H, true);
    float *dXi = matrixMultiTrans(dhi, 1, H, false, cache.W_i, Z, H, true);
    float *dXo = matrixMultiTrans(dho, 1, H, false, cache.W_o, Z, H, true);
    float *dXc = matrixMultiTrans(dhc, 1, H, false, cache.W_o, Z, H, true);
    
    for (int i = 0; i < H; i++) {
        dh_next[i] = dXf[i] + dXc[i] + dXi[i] + dXo[i];
        dc_next[i] = cache.hf[i] * dc[i];
    }
}
