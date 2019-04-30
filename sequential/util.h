#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#define indexNorm(i, j, w) (i * w + j)

void softmax(float *input, int input_len);

float sigmoid_forward(float x);

float dsigmoid(float input);

float dtanh(float input);

//static inline float *dsigmoid_vector(float *input, int len);
//
//static inline float *dtanh_vector(float *input, int len);

//static inline int indexNormal(int i, int j, int w);

int indexTrans(int i, int j, int height, int width, bool isTrans);

float *matrixMulti(float *X, int X_w, int X_h, float *Y, int Y_w, int Y_h);

float *matrixMultiTrans(float *X, int X_h, int X_w, bool X_isTrans, float *Y, int Y_h, int Y_w, bool Y_isTrans);

void showWeights(float *X, int lenX, char *name);

void initValues(float *input, int lenX, float value);