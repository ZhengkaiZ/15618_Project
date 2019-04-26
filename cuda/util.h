#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#define index(i, j, w) (i * w + j)

static void softmax(float *input, int input_len);

static float sigmoid_forward(float x);

static float dsigmoid(float input);

static float dtanh(float input);

static float *dsigmoid_vector(float *input, int len);

static float *dtanh_vector(float *input, int len);

static int indexTrans(int i, int j, int height, int width, bool isTrans);

static float *matrixMulti(float *X, int X_w, int X_h, float *Y, int Y_w, int Y_h);

static float *matrixMultiTrans(float *X, int X_h, int X_w, bool X_isTrans, float *Y, int Y_h, int Y_w, bool Y_isTrans);

static void showWeights(float *X, int lenX, char *name);
