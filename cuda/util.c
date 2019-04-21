#include "util.h"

static inline void softmax_forward(float *input, int input_len) {
    int i;
    float m;
    /* Find maximum value from input array */
    m = input[0];
    for (i = 1; i < input_len; i++) {
        if (input[i] > m) {
            m = input[i];
        }
    }

    float sum = 0;
    for (i = 0; i < input_len; i++) {
        sum += expf(input[i]-m);
    }

    for (i = 0; i < input_len; i++) {
        input[i] = expf(input[i] - m - log(sum));
    }
}

static inline void softmax_backword() {

}

static inline float sigmoid_forward(float x) {
    float exp_value;
    float return_value;

    exp_value = exp((double) -x);
    return_value = 1 / (1 + exp_value);

    return return_value;
}

static inline float sigmoid_backward(float x) {

}

static inline double tanh_forward(double x) {
    return tanh(x);
}

static inline float tanh_backward() {

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
