#ifndef UTIL_H
/*
  C library implementing helper math and timing functions.
*/

#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define index(i, j, w) (i * w + j)

static inline void softmax_forward(float *input, int input_len);

static inline float sigmoid_forward(float x);

static float* matrixMulti(float *X, int X_w, int X_h, float *Y, int Y_w, int Y_h);

#define UTIL_H
#endif 