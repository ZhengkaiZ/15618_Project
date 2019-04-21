#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include "struct.h"

#define index(i, j, w) (i * w + j)

static inline void softmax(float *input, int input_len);

static inline float sigmoid_forward(float x);

static inline float dsigmoid(float input);

static inline float dtanh(float input);

static inline float *dsigmoid_vector(float *input, int len);

static inline float *dtanh_vector(float *input, int len);

static inline int indexTrans(int i, int j, int height, int width, bool isTrans);

static float *matrixMulti(float *X, int X_w, int X_h, float *Y, int Y_w, int Y_h);

static float *matrixMultiTrans(float *X, int X_h, int X_w, bool X_isTrans, float *Y, int Y_h, int Y_w, bool Y_isTrans);
