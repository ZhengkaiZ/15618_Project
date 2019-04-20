#include "struct.h"

static void initiateModel(Model model) {
    model.W_f = (float*)malloc(sizeof(float) * Z * H);
    model.W_i = (float*)malloc(sizeof(float) * Z * H);
    model.W_c = (float*)malloc(sizeof(float) * Z * H);
    model.W_o = (float*)malloc(sizeof(float) * Z * H);
    model.W_y = (float*)malloc(sizeof(float) * H * D);

    model.b_f = (float*)malloc(sizeof(float) * H);
    model.b_i = (float*)malloc(sizeof(float) * H);
    model.b_c = (float*)malloc(sizeof(float) * H);
    model.b_o = (float*)malloc(sizeof(float) * H);
    model.b_y = (float*)malloc(sizeof(float) * D);
}

static State getNewtate() {
    State state;
    state.h_f = (float*)malloc(sizeof(float) * H);
    state.h_i = (float*)malloc(sizeof(float) * H);
    state.h_c = (float*)malloc(sizeof(float) * H);
    state.h_o = (float*)malloc(sizeof(float) * H);

    state.h = (float*)malloc(sizeof(float) * H);
    state.c = (float*)malloc(sizeof(float) * H);
    return state;
}
