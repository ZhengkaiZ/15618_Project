#include "util.h"

void softmax(float *input, int input_len) {
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
        sum += expf(input[i] - m);
    }

    for (i = 0; i < input_len; i++) {
        input[i] = expf(input[i] - m - log(sum));
    }
}

float sigmoid_forward(float x) {
    float exp_value;
    float return_value;

    exp_value = exp((double) -x);
    return_value = 1 / (1 + exp_value);

    return return_value;
}

//static float sigmoid_backward(float x) {
//    return x * (1 - x);
//}
//
//static float tanh_forward(float x) {
//    return tanhf(x);
//}
//
//static float tanh_backward(float x) {
//    return 1 - x * x;
//}

float dsigmoid(float input) {
    return input * (1 - input);
}

float dtanh(float input) {
    return 1 - input * input;
}

//static float *dsigmoid_vector(float *input, int len) {
//    int i;
//    float *output = (float *) malloc(len * sizeof(float));
//    for (i = 0; i < len; i++) {
//        output[i] = dsigmoid(input[i]);
//    }
//
//    return output;
//}

//static inline float *dtanh_vector(float *input, int len) {
//    int i = 0;
//    float *output = (float *) malloc(len * sizeof(float));
//    for (i = 0; i < len; i++) {
//        output[i] = dtanh(input[i]);
//    }
//
//    return output;
//}


inline float *matrixMulti(float *X, int X_w, int X_h, float *Y, int Y_w, int Y_h) {
    float *result = (float *) malloc(sizeof(float) * X_w * Y_h);
    for (int i = 0; i < X_w; i++) {
        for (int j = 0; j < Y_h; j++) {

            int index_c = indexNorm(i, j, X_h);
            result[index_c] = 0;
            for (int k = 0; k < X_h; k++) {
                int index_a = indexNorm(i, k, X_h);
                int index_b = indexNorm(k, j, Y_h);
                result[index_c] += X[index_a] * Y[index_b];
            }

        }
    }

    return result;
}


int indexTrans(int i, int j, int height, int width, bool isTrans) {
    if (isTrans) {
        return j * height + i;
    } else {
        return i * width + j;
    }
}

float *matrixMultiTrans(float *X, int X_h, int X_w, bool X_isTrans, float *Y, int Y_h, int Y_w, bool Y_isTrans) {
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

    float *result = (float *) malloc(sizeof(float) * Xh * Yw);
    for (int i = 0; i < Xh; i++) {
        for (int j = 0; j < Yw; j++) {
            int index_c = indexTrans(i, j, Xh, Yw, false);
            float tmp = 0;
            for (int k = 0; k < Xw; k++) {
                int index_a = indexTrans(i, k, Xh, Xw, X_isTrans);
                int index_b = indexTrans(k, j, Yh, Yw, Y_isTrans);
                tmp += X[index_a] * Y[index_b];
            }
            result[index_c] = tmp;
        }
    }
    return result;
}

//static void showWeights(float *X, int lenX, char *name) {
//    printf("The weights of %s is: \n", name);
//    for (int i = 0; i < lenX; i++) {
//        printf("%f ", X[i]);
//    }
//    printf("\n");
//}

