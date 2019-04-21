#ifndef CUDA_STRUCT_H

#include "util.h"
#include "param.h"

typedef struct {
    // Weights
    float *W_f; // (D + H) x H
    float *W_i; // (D + H) x H
    float *W_c; // (D + H) x H
    float *W_o; // (D + H) x H
    float *W_y; // H x D

    // Bias
    float *b_f; // 1 x H
    float *b_i; // 1 x H
    float *b_c; // 1 x H
    float *b_o; // 1 x H
    float *b_y; // 1 x D
}Model;

typedef struct {
    // Hidden State
    float *h_f;
    float *h_i;
    float *h_c;
    float *h_o;

    // Current State
    float *h;
    float *c;
}State;

static void initiateModel();

static State getNewtate();
