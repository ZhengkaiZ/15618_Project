#include "util.h"

#define H 128
#define D 10
#define Z 138

typedef struct {
    float *W_f;
    float *W_i;
    float *W_c;
    float *W_o;
    float *W_y;

    float *b_f;
    float *b_i;
    float *b_c;
    float *b_o;
    float *b_y;
}Model;

Model model;

static void initiateModel() {
    model.W_f = (float*)malloc(sizeof(float) * Z * H);
    model.W_i = (float*)malloc(sizeof(float) * Z * H);
    model.W_c = (float*)malloc(sizeof(float) * Z * H);
    model.W_o = (float*)malloc(sizeof(float) * Z * H);
    model.W_y = (float*)malloc(sizeof(float) * H * D);

    model.b_f = (float*)malloc(sizeof(float) * H);
    model.b_i = (float*)malloc(sizeof(float) * H);
    model.b_c = (float*)malloc(sizeof(float) * H);
    model.b_o = (float*)malloc(sizeof(float) * H);
    model.b_y = (float*)malloc(sizeof(float) * D);
}

static inline int index(int i, int j, int width) {
    return i * width + j;
}

static float* matrixMulti(float *X, int X_w, int X_h, float *Y, int Y_w, int Y_h) {
    float* result = (float*)malloc(sizeof(float) * X_w * Y_h);
    for (int i = 0; i < X_w; i++) {
        for (int j = 0; j < Y_h; j++) {

            int index_c = index(i, j, X_w);
            result[index_c] = 0;
            for (int k = 0; k < X_h; k++) {
                int index_a = index(i, k, X_w);
                int index_b = index(k, j, Y_w);
                result[index_c] += X[index_a] * Y[index_b];
            }

        }
    }

    return result;
}

static float* forward(int* input, float* h, float* c) {
    float *W_f = model.W_f;
    float *W_i = model.W_i;
    float *W_c = model.W_c;
    float *W_o = model.W_o;
    float *W_y = model.W_y;

    float *b_f = model.b_f;
    float *b_i = model.b_i;
    float *b_c = model.b_c;
    float *b_o = model.b_o;
    float *b_y = model.b_y;

    // One-hot encode
    float *X_one_hot = (float*)malloc(sizeof(float) * D);
    for (int i = 0; i < Z * H; i++) {
        X_one_hot[input[i]] = 1.0;
    }
    float* combined_input = (float*)malloc(sizeof(float) * (H + D));
    for (int i = 0; i < H; i++) {
        combined_input[i] = h[i];
    }
    for (int i = H; i < Z; i++) {
        combined_input[i] = X_one_hot[i - H];
    }

    float *h_f = (float*)malloc(sizeof(float) * H);
    float *h_i = (float*)malloc(sizeof(float) * H);
    float *h_c = (float*)malloc(sizeof(float) * H);
    float *h_o = (float*)malloc(sizeof(float) * H);

    float* temp = matrixMulti(combined_input, 2, Z, W_f, Z, H);
    for (int i = 0; i < H; ++i) {
        h_f[i] = sigmoid(temp[i] + b_f[i]);
    }

    temp = matrixMulti(combined_input, 1, Z, W_i, Z, H);
    for (int i = 0; i < H; ++i) {
        h_i[i] = sigmoid(temp[i] + b_i[i]);
    }

    temp = matrixMulti(combined_input, 1, Z, W_o, Z, H);
    for (int i = 0; i < H; ++i) {
        h_o[i] = sigmoid(temp[i] + b_o[i]);
    }

    temp = matrixMulti(combined_input, 1, Z, W_c, Z, H);
    for (int i = 0; i < H; ++i) {
        h_c[i] = tanhf(temp[i] + b_c[i]);
    }

    for (int i = 0; i < H; i++) {
        c[i] = h_f[i] * c[i] + h_i[i] * h_c[i];
        h[i] = h_o[i] * tanhf(c[i]);
    }

    float *y = (float*)malloc(sizeof(float) * D);
    temp = matrixMulti(h, 1, H,W_y, H, D);
    for (int i = 0; i < D; i++) {
        y[i] = temp[i] + b_y[i];
    }

    softmax(y, D);

    return y;
}