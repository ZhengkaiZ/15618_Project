#include "struct.h"

static Model getNewModel() {
    Model model;
    model.W_f = (float *) malloc(sizeof(float) * Z * H);
    model.W_i = (float *) malloc(sizeof(float) * Z * H);
    model.W_c = (float *) malloc(sizeof(float) * Z * H);
    model.W_o = (float *) malloc(sizeof(float) * Z * H);
    model.W_y = (float *) malloc(sizeof(float) * H * D);

    model.b_f = (float *) malloc(sizeof(float) * H);
    model.b_i = (float *) malloc(sizeof(float) * H);
    model.b_c = (float *) malloc(sizeof(float) * H);
    model.b_o = (float *) malloc(sizeof(float) * H);
    model.b_y = (float *) malloc(sizeof(float) * D);
    return model;
}

static State getNewState() {
    State state;
    state.h = (float *) malloc(sizeof(float) * H);
    state.c = (float *) malloc(sizeof(float) * H);
    return state;
}

static HiddenState getNewHiddenState() {
    HiddenState hiddenState;
    hiddenState.h_f = (float *) malloc(sizeof(float) * H);
    hiddenState.h_i = (float *) malloc(sizeof(float) * H);
    hiddenState.h_c = (float *) malloc(sizeof(float) * H);
    hiddenState.h_o = (float *) malloc(sizeof(float) * H);

    hiddenState.X = (float *) malloc(sizeof(float) * Z);
    return hiddenState;
}

static void updateModel(Model *model, Model *grad) {
    for(int i = 0; i < (Z*H); i++) {
        model->W_f[i] += grad->W_f[i];
        model->W_i[i] += grad->W_i[i];
        model->W_c[i] += grad->W_c[i];
        model->W_o[i] += grad->W_o[i];
    }

    for(int i = 0; i < H*D; i++) {
        model->W_y[i] += grad->W_y[i];
    }

    for(int i = 0; i < H; i++) {
        model->b_f[i] += grad->b_f[i];
        model->b_i[i] += grad->b_i[i];
        model->b_c[i] += grad->b_c[i];
        model->b_o[i] += grad->b_o[i];
    }

    for(int i = 0; i < D; i++) {
        model->b_y[i] += grad->b_y[i];
    }
}

