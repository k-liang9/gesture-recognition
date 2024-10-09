#include "Dense.h"
#include <Eigen/Dense>
#include <cassert>
#include <random>

using Eigen::VectorXd;
using Eigen::MatrixXd;

::Dense::Dense(MatrixXd *w, VectorXd *b, activation a) :
        weights{*w},
        biases{*b} {
    set_activation_func(a);
    dropout_mask(biases.size());
    activations(biases.size());
    assert(weights.rows() == weights.size());
}

void ::Dense::propagate(MatrixXd& prev_activations) {
    assert(prev_activations.size() == weights.cols());
    activations = weights*prev_activations-biases;
}

void ::Dense::dropout(float dropout_rate) {
    int expected_rate = static_cast<int>(100*dropout_rate);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 99);

    int actual_rate = 0;
    for (int i = 0; i < activations.size(); ++i) {
        if (dis(gen) < expected_rate) {
            dropout_mask[i] = 0;
            actual_rate++;
        } else {
            dropout_mask[i] = 1;
        }
    }

    float scale = 1.0f/((100.0f-actual_rate)/100.0f);
    for (int i = 0; i < activations.size(); ++i) {
        if (dropout_mask[i] == 0) {
            activations[i] = 0;
        } else {
            activations[i] *= scale;
        }
    }
}