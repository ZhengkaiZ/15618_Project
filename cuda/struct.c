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

static State getNewtate() {
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

