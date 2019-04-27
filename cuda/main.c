#include "util.h"
#include "param.h"
#include "struct.h"
#include <string.h>

Model model;
static void cell_forward(int input, State *old_state, State *state, HiddenState *h, float *prob, int layer);
static void cell_backward(Model *grad, float **prob, int y_train, State *old_state, State *state, State *new_state, HiddenState hiddenState, int layer, int t);
void train(int *X, int *Y, Model *grad);
void SGD(int **X, int **Y, State *state, float learning_rate, int num_samples);

static void cell_forward(int input, State *old_state, State *state, HiddenState *h, float *prob, int layer) {
    float *W_f = model.W_f;
    float *W_i = model.W_i;
    float *W_c = model.W_c;
    float *W_o = model.W_o;
    float *W_y = model.W_y;

    float *b_f = model.b_f;
    float *b_i = model.b_i;
    float *b_c = model.b_c;
    float *b_o = model.b_o;
    float *b_y = model.b_y;

    // One-hot encode
    float *X_one_hot;
    if (layer == 1) {
        X_one_hot = (float *) malloc(sizeof(float) * D);
        X_one_hot[input] = 1.0;
    } else {
        X_one_hot = prob;
    }

    // Combine input
    for (int i = 0; i < H; i++) {
        h->X[i] = old_state->h[i];
    }

    for (int i = H; i < Z; i++) {
        h->X[i] = X_one_hot[i - H];
    }


    // Forget Gate
    // hf = sigmoid(X @ Wf + bf)
    float *temp = matrixMulti(h->X, 1, Z, W_f, Z, H);
    for (int i = 0; i < H; ++i) {
        h->h_f[i] = sigmoid_forward(temp[i] + b_f[i]);
    }

    // Input Gate
    // hi = sigmoid(X @ Wi + bi)
    temp = matrixMulti(h->X, 1, Z, W_i, Z, H);
    for (int i = 0; i < H; ++i) {
        h->h_i[i] = sigmoid_forward(temp[i] + b_i[i]);
    }

//    // Detecting input pattern
//    // hc = tanh(X @ Wc + bc)
    temp = matrixMulti(h->X, 1, Z, W_c, Z, H);
    for (int i = 0; i < H; ++i) {
        h->h_c[i] = tanhf(temp[i] + b_c[i]);
    }
    // Output Gate
    // ho = sigmoid(X @ Wo + bo)
    temp = matrixMulti(h->X, 1, Z, W_o, Z, H);
    for (int i = 0; i < H; ++i) {
        h->h_o[i] = sigmoid_forward(temp[i] + b_o[i]);
    }

    // c = hf * c_old + hi * hc
    // h = ho * tanh(c)
    for (int i = 0; i < H; i++) {
        state->c[i] = h->h_f[i] * old_state->c[i] + h->h_i[i] * h->h_c[i];
        state->h[i] = h->h_o[i] * tanhf(state->c[i]);
    }

    // y = h @ Wy + by
    temp = matrixMulti(state->h, 1, H, W_y, H, D);
    for (int i = 0; i < D; i++) {
        prob[i] = temp[i] + b_y[i];
    }

    // prob = softmax(y)
    softmax(prob, D);
}

static void cell_backward(Model *grad, float **prob, int y_train, State *old_state, State *state, State *new_state,
                          HiddenState hiddenState, int layer, int t) {

    float *dh_next = new_state->h;
    float *dc_next = new_state->c;

//    // Softmax loss gradient
    float *dy;
    if (layer == LAYER - 1) {
        dy = (float *) malloc(sizeof(float) * D);
        memcpy(dy, prob, D);
        dy[y_train] -= 1.0;
    } else {   // Problem
        dy = prob[t];
    }

    // Hidden to output gradient
    grad->W_y = matrixMultiTrans(state->h, 1, H, true, dy, 1, D, false);
    grad->b_y = dy;
    float *dh = matrixMultiTrans(dy, 1, D, false, model.W_y, H, D, true);
    for (int i = 0; i < H; i++) {
        dh[i] += dh_next[i];
    }

    // Gradient for h_o in
    // h = h_o * tanh(c)
    float *dho = (float *) malloc(H * sizeof(float));
    for (int i = 0; i < H; i++) {
        dho[i] = tanhf(state->c[i]) * dh[i] * dsigmoid(hiddenState.h_o[i]);
    }

    // Gradient for c in
    // h = h_o * tanh(c)
    float *dc = (float *) malloc(H * sizeof(float));
    for (int i = 0; i < H; i++) {
        dc[i] = hiddenState.h_o[i] * dh[i] * dtanh(state->c[i]) + dc_next[i];
    }

    // Gradient for h_f in
    // c = h_f * c_old + h_i * h_c
    float *dhf = (float *) malloc(H * sizeof(float));
    for (int i = 0; i < H; i++) {
        dhf[i] = old_state->c[i] * dc[i] * dsigmoid(hiddenState.h_f[i]);
    }

    // Gradient for h_i in
    // c = h_f * c_old + h_i * h_c
    float *dhi = (float *) malloc(H * sizeof(float));
    for (int i = 0; i < H; i++) {
        dhi[i] = hiddenState.h_c[i] * dc[i] * dsigmoid(hiddenState.h_i[i]);
    }

//    // Gradient for h_c in
//    // c = h_f * c_old + h_i * h_c
    float *dhc = (float *) malloc(H * sizeof(float));
    for (int i = 0; i < H; i++) {
        dhc[i] = hiddenState.h_i[i] * dc[i] * dtanh(hiddenState.h_c[i]);
    }

    // Gate gradients
    grad->W_f = matrixMultiTrans(hiddenState.X, 1, Z, true, dhf, 1, H, false);
    grad->b_f = dhf;

    grad->W_i = matrixMultiTrans(hiddenState.X, 1, Z, true, dhi, 1, H, false);
    grad->b_i = dhi;

    grad->W_o = matrixMultiTrans(hiddenState.X, 1, Z, true, dho, 1, H, false);
    grad->b_o = dho;

    grad->W_c = matrixMultiTrans(hiddenState.X, 1, Z, true, dhc, 1, H, false);
    grad->b_c = dhc;

    float *dXf = matrixMultiTrans(dhf, 1, H, false, model.W_f, Z, H, true);
    float *dXi = matrixMultiTrans(dhi, 1, H, false, model.W_i, Z, H, true);
    float *dXo = matrixMultiTrans(dho, 1, H, false, model.W_o, Z, H, true);
    float *dXc = matrixMultiTrans(dhc, 1, H, false, model.W_c, Z, H, true);
//
    for (int i = H; i < Z; i++) {
        prob[t][i-H] = dXf[i-H] + dXc[i-H] + dXi[i-H] + dXo[i-H];
    }
//
    for (int i = 0; i < H; i++) {
        dh_next[i] = dXf[i] + dXc[i] + dXi[i] + dXo[i];
        dc_next[i] = hiddenState.h_f[i] * dc[i];
    }

}

void train(int *X, int *Y, Model *grad) {
    // Forward
    State *states = (State *) malloc(sizeof(State) * (TIME + 1) * (LAYER + 1));
    HiddenState *hiddenState = (HiddenState *) malloc(sizeof(HiddenState) * (TIME + 1) * (LAYER + 1));
    float **probs = (float **) malloc(sizeof(float *) * TIME);
    for (int i = 0; i < LAYER; i++) {
        states[i] = getNewState();
        hiddenState[i] = getNewHiddenState();
    }

    int i = LAYER;
    for (int t = 1; t <= TIME; t++) {
        states[i] = getNewState();
        hiddenState[i] = getNewHiddenState();
        probs[t - 1] = (float *) malloc(sizeof(float) * D);

        memcpy(states[i].h, &X[t * D], H);    // Initialize h(0) to input
        memset(states[i].c, 0, H);          // Initialize c(0) to 0
        i++;
        // Hidden Layer operate at time t
        for (int l = 1; l <= LAYER; l++) {
            states[i] = getNewState();
            hiddenState[i] = getNewHiddenState();
            cell_forward(X[t * D], &states[i - LAYER], &states[i], &hiddenState[i], probs[t - 1], l);
            i++;
        }
    }

    printf("%d ", i);
    // Backward
    // Gradient for dh_next and dc_next is zero from the last t
    State d_next[LAYER];
    for (int k = 0; k < LAYER; k++) {
        d_next[k] = getNewState();
        memset(d_next[k].h, 0, H);
        memset(d_next[k].c, 0, H);
    }

    for (int t = TIME; t >= 1; t--) {
        for (int l = LAYER - 1; l >= 0; l--) {
            cell_backward(grad, probs, Y[t - 1], &states[(t - 1) * (LAYER + 1) + l], &states[t * (LAYER + 1) + l], &d_next[l], hiddenState[t * (LAYER + 1) + l], l, t - 1);
        }
    }
}

void SGD(int **X, int **Y, State *state, float learning_rate, int num_samples) {
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < num_samples; j++) {
            Model tmp_grad = getNewModel();
            train(X[j], Y[j], &tmp_grad);
            updateModel(&model, &tmp_grad, learning_rate);
        }
    }
}

int main(void) {
    State state = getNewState();
    int learning_rate = 1;
    int num_samples = 5;
    model = getNewModel();


    int **X = (int **) malloc(num_samples * sizeof(int*));
    int **Y = (int **) malloc(num_samples * sizeof(int*));
    for (int i = 0; i < num_samples; i++) {
        X[i] = (int*) calloc(TIME, sizeof(int));
        Y[i] = (int*) calloc(TIME, sizeof(int));
    }
    SGD(X, Y, &state, learning_rate, num_samples);

    return 1;
}
