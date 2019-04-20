#ifndef UTIL_H
/*
  C library implementing helper math and timing functions.
*/

#include <stdlib.h>
#include <stdint.h>
#include <math.h>

static inline void softmax_forward(float *input, int input_len);

static inline float sigmoid_forward(float x);

static inline int index(int i, int j, int width);

static float* matrixMulti(float *X, int X_w, int X_h, float *Y, int Y_w, int Y_h);

#define UTIL_H
#endif 