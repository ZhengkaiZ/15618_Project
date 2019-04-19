#include "util.h"

static inline void softmax(float *input, int input_len) {
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

static inline float sigmoid(float x) {
    float exp_value;
    float return_value;

    exp_value = exp((double) -x);
    return_value = 1 / (1 + exp_value);

    return return_value;
}
