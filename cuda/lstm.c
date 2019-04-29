#include "struct.h"
#include "util.h"

Model getNewModel() {
    Model model;
    model.W_f = (float *) malloc(sizeof(float) * Z * H);
    initValues(model.W_f, Z * H, 0.1);
    model.W_i = (float *) malloc(sizeof(float) * Z * H);
    initValues(model.W_i, Z * H, 0.1);
    model.W_c = (float *) malloc(sizeof(float) * Z * H);
    initValues(model.W_c, Z * H, 0.1);
    model.W_o = (float *) malloc(sizeof(float) * Z * H);
    initValues(model.W_o, Z * H, 0.1);
    model.W_y = (float *) malloc(sizeof(float) * H * D);
    initValues(model.W_y, H * D, 0.1);

    model.b_f = (float *) malloc(sizeof(float) * H);
    initValues(model.b_f, H, 0.0);
    model.b_i = (float *) malloc(sizeof(float) * H);
    initValues(model.b_i, H, 0.0);
    model.b_c = (float *) malloc(sizeof(float) * H);
    initValues(model.b_c, H, 0.0);
    model.b_o = (float *) malloc(sizeof(float) * H);
    initValues(model.b_o, H, 0.0);
    model.b_y = (float *) malloc(sizeof(float) * D);
    initValues(model.b_y, D, 0.0);
    return model;
}

State getNewState() {
    State state;
    state.h = (float *) calloc(H, sizeof(float));
    initValues(state.h, H, 0.0);
    state.c = (float *) calloc(H, sizeof(float));
    initValues(state.c, H, 0.0);
    state.dX = (float *) calloc(Z, sizeof(float));
    initValues(state.dX, Z, 0.0);
    return state;
}

HiddenState getNewHiddenState() {
    HiddenState hiddenState;
    hiddenState.h_f = (float *) malloc(sizeof(float) * H);
    hiddenState.h_i = (float *) malloc(sizeof(float) * H);
    hiddenState.h_c = (float *) malloc(sizeof(float) * H);
    hiddenState.h_o = (float *) malloc(sizeof(float) * H);

    hiddenState.X = (float *) malloc(sizeof(float) * Z);

    return hiddenState;
}

void updateModel(Model *model, Model *grad, float learning_rate) {
    for (int i = 0; i < (Z*H); i++) {
        model->W_f[i] -= learning_rate * grad->W_f[i];
        model->W_i[i] -= learning_rate * grad->W_i[i];
        model->W_c[i] -= learning_rate * grad->W_c[i];
        model->W_o[i] -= learning_rate * grad->W_o[i];
    }

    for (int i = 0; i < H*D; i++) {
        model->W_y[i] -= learning_rate * grad->W_y[i];
    }

    for (int i = 0; i < H; i++) {
        model->b_f[i] -= learning_rate * grad->b_f[i];
        model->b_i[i] -= learning_rate * grad->b_i[i];
        model->b_c[i] -= learning_rate * grad->b_c[i];
        model->b_o[i] -= learning_rate * grad->b_o[i];
    }

    for (int i = 0; i < D; i++) {
        model->b_y[i] -= learning_rate * grad->b_y[i];
    }
}

void updateGrad(Model *grad, Model *tmp_grad) {
    for (int i = 0; i < (Z*H); i++) {
        grad->W_f[i] += tmp_grad->W_f[i];
        grad->W_i[i] += tmp_grad->W_i[i];
        grad->W_c[i] += tmp_grad->W_c[i];
        grad->W_o[i] += tmp_grad->W_o[i];
    }
    
    for (int i = 0; i < H*D; i++) {
        grad->W_y[i] += tmp_grad->W_y[i];
    }
    
    for (int i = 0; i < H; i++) {
        grad->b_f[i] += tmp_grad->b_f[i];
        grad->b_i[i] += tmp_grad->b_i[i];
        grad->b_c[i] += tmp_grad->b_c[i];
        grad->b_o[i] += tmp_grad->b_o[i];
    }
    
    for (int i = 0; i < D; i++) {
        grad->b_y[i] += tmp_grad->b_y[i];
    }
}
