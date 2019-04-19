#ifndef UTIL_H
/*
  C library implementing helper math and timing functions.
*/

#include <stdlib.h>
#include <stdint.h>
#include <math.h>

static inline void softmax(float *input, int input_len);

static inline float sigmoid(float x);

#define UTIL_H
#endif 