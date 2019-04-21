#include "util.h"
#include "struct.h"
#include "param.h"

Model model;

static void cell_forward(int* input, float* old_h, float* old_c, State* state, float* prob) {
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
    float *X_one_hot = (float*)malloc(sizeof(float) * D);
    for (int i = 0; i < Z * H; i++) {
        X_one_hot[input[i]] = 1.0;
    }

    // Combine input
    float* combined_input = (float*)malloc(sizeof(float) * (H + D));
    for (int i = 0; i < H; i++) {
        combined_input[i] = old_h[i];
    }
    for (int i = H; i < Z; i++) {
        combined_input[i] = X_one_hot[i - H];
    }

    // Forget Gate
    // hf = sigmoid(X @ Wf + bf)
    float* temp = matrixMulti(combined_input, 2, Z, W_f, Z, H);
    for (int i = 0; i < H; ++i) {
        state->h_f[i] = sigmoid_forward(temp[i] + b_f[i]);
    }

    // Input Gate
    // hi = sigmoid(X @ Wi + bi)
    temp = matrixMulti(combined_input, 1, Z, W_i, Z, H);
    for (int i = 0; i < H; ++i) {
        state->h_i[i] = sigmoid_forward(temp[i] + b_i[i]);
    }

    // Detecting input pattern
    // hc = tanh(X @ Wc + bc)
    temp = matrixMulti(combined_input, 1, Z, W_c, Z, H);
    for (int i = 0; i < H; ++i) {
        state->h_c[i] = tanhf(temp[i] + b_c[i]);
    }

    // Output Gate
    // ho = sigmoid(X @ Wo + bo)
    temp = matrixMulti(combined_input, 1, Z, W_o, Z, H);
    for (int i = 0; i < H; ++i) {
        state->h_o[i] = sigmoid_forward(temp[i] + b_o[i]);
    }

    // c = hf * c_old + hi * hc
    // h = ho * tanh(c)
    for (int i = 0; i < H; i++) {
        state->c[i] = state->h_f[i] * old_c[i] + state->h_i[i] * state->h_c[i];
        state->h[i] = state->h_o[i] * tanhf(state->c[i]);
    }

    // y = h @ Wy + by
    temp = matrixMulti(state->h, 1, H,W_y, H, D);
    for (int i = 0; i < D; i++) {
        prob[i] = temp[i] + b_y[i];
    }

    // prob = softmax(y)
    softmax_forward(prob, D);
}

void forward(int* X, float* Y) {
    State* states = (State*)malloc(sizeof(State) * TIME * LAYER);
    float** probs = (float**)malloc(sizeof(float*) * TIME * LAYER);

    for (int t = 0; t < TIME; t++) {
        states[t * LAYER] = getNewtate();
        states[t * LAYER].h = X[t * D];
        states[t * LAYER].c = 0;
        probs[t] = (float*)malloc(sizeof(float) * D);
        for (int l = 1; l < LAYER; l++) {
            states[t * LAYER + l] = getNewtate();
            cell_forward(&X[t * D], states[t * LAYER + l].h, states[t * LAYER + l].c, &states[t * LAYER + l], probs[t]);
        }
    }
}

float* cell_backward(float *prob, State *state) {

}

void *train(int* X, int* Y, State* state, int time) {
    float loss;

    // Forward Step


    // The loss is the average cross entropy
    //    loss /= X_train.shape[0]


    // Backward Step

    /*
    # Gradient for dh_next and dc_next is zero for the last timestep
    d_next = (np.zeros_like(h), np.zeros_like(c))
    grads = {k: np.zeros_like(v) for k, v in model.items()}

    # Go backward from the last timestep to the first
    for prob, y_true, cache in reversed(list(zip(probs, y_train, caches))):
        grad, d_next = lstm_backward(prob, y_true, d_next, cache)

        # Accumulate gradients from all timesteps
        for k in grads.keys():
            grads[k] += grad[k]

    return grads, loss, state
     */
}

void SDG()