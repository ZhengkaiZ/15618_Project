//#include <stdlib.h>
//#include "param.h"

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
} Model;

typedef struct {
    float *c;   // len = H, Current value of CEC
    float *h;   // len = H, Current hidden state value
    
    float *dX;
} State;

typedef struct {
    // Hidden State
    float *h_f; // len = H, Forget Gate
    float *h_i; // len = H, Input Gate
    float *h_c; // len = H, Detecting input pattern
    float *h_o; // len = H, Output Gate

    float *X;     // Combined input of that layer
} HiddenState;

static Model getNewModel();

static State getNewState();

static HiddenState getNewHiddenState();

static void updateModel(Model *model, Model *grad, float learning_rate);
static void updateGrad(Model *grad, Model *tmp_grad);
