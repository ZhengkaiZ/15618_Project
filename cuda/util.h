#ifndef UTIL_H
/*
  C library implementing helper math and timing functions.
*/

#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define H 128
#define D 10
#define Z 138

typedef struct {
    float *W_f;
    float *W_i;
    float *W_c;
    float *W_o;
    float *W_y;
    
    float *b_f;
    float *b_i;
    float *b_c;
    float *b_o;
    float *b_y;
} Model;

typedef struct {
    float *X;
    float *h;
    float *c;
    float *c_old;
    
    float *ho;
    float *hf;
    float *hc;
    float *hi;
    
} Cache;

Model model;

Model grad;

Cache cache;

static inline void softmax(float *input, int input_len);

static inline float sigmoid(float x);

static inline float dsigmoid(float input);

static inline float dtanh(float input);

static inline float* dsigmoid_vector(float *input, int len);

static inline float* dtanh_vector(float *input, int len);

static inline float* deep_copy(int* input, int input_len);

static inline int index(int i, int j, int width);

static inline int indexTrans(int i, int j, int height, int width, bool isTrans);

static float* matrixMulti(float *X, int X_w, int X_h, float *Y, int Y_w, int Y_h);

static float* matrixMultiTrans(float *X, int X_h, int X_w, bool X_isTrans, float *Y, int Y_h, int Y_w, bool Y_isTrans);

static inline float* deep_copy(int* input, int input_len);

static Model backward(float* prob, int y_train, float* d_next_h, float* d_next_c);

static float* forward(int* input, float* h, float* c);

static Model backward(float* prob, int y_train, float* d_next_h, float* d_next_c);


#define UTIL_H
#endif 
