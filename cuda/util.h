#ifndef UTIL_H
/*
  C library implementing helper math and timing functions.
*/

#include <stdlib.h>
#include <stdint.h>
#include <math.h>

static float softmax(float *input, int input_len);

static float sigmoid(float x);

#define UTIL_H
#endif 