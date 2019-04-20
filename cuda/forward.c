#include "util.h"
#include "struct.h"


Model model;

static float* forward(int* input, float* old_h, float* old_c, State* state) {
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
        combined_input[i] = old_h[i];
    }
    for (int i = H; i < Z; i++) {
        combined_input[i] = X_one_hot[i - H];
    }

    // hf = sigmoid(X @ Wf + bf)
    float* temp = matrixMulti(combined_input, 2, Z, W_f, Z, H);
    for (int i = 0; i < H; ++i) {
        state->h_f[i] = sigmoid_forward(temp[i] + b_f[i]);
    }

    // hi = sigmoid(X @ Wi + bi)
    temp = matrixMulti(combined_input, 1, Z, W_i, Z, H);
    for (int i = 0; i < H; ++i) {
        state->h_i[i] = sigmoid_forward(temp[i] + b_i[i]);
    }

    // ho = sigmoid(X @ Wo + bo)
    temp = matrixMulti(combined_input, 1, Z, W_o, Z, H);
    for (int i = 0; i < H; ++i) {
        state->h_o[i] = sigmoid_forward(temp[i] + b_o[i]);
    }

    // hc = tanh(X @ Wc + bc)
    temp = matrixMulti(combined_input, 1, Z, W_c, Z, H);
    for (int i = 0; i < H; ++i) {
        state->h_c[i] = tanhf(temp[i] + b_c[i]);
    }

    // c = hf * c_old + hi * hc
    // h = ho * tanh(c)
    for (int i = 0; i < H; i++) {
        state->c[i] = state->h_f[i] * old_c[i] + state->h_i[i] * state->h_c[i];
        state->h[i] = state->h_o[i] * tanhf(state->c[i]);
    }

    // y = h @ Wy + by
    float *y = (float*)malloc(sizeof(float) * D);
    temp = matrixMulti(state->h, 1, H,W_y, H, D);
    for (int i = 0; i < D; i++) {
        y[i] = temp[i] + b_y[i];
    }

    // prob = softmax(y)
    softmax_forward(y, D);

    return y;
}

float* backward(float *prob, State *state) {

}

void *train(int epoch) {
    for (int e = 0; e < epoch; e++) {

    }
}