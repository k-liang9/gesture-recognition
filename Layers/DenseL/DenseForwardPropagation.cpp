#include "DenseL.h"
#include "../Layer.h"
#include <Eigen/Dense>
#include <cassert>
#include <random>
#include <iostream>

using namespace Eigen;

void DenseL::propagate() {
    assert(get_prev_layer()->get_activations().size() == weights.cols());
    get_activations() = weights * get_prev_layer()->get_activations() - biases;
}

void DenseL::dropout(float dropout_rate) {
    VectorXd& activations = get_activations();
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

    dropout_used_count += dropout_mask;

    float scale = 1.0f/((100.0f-actual_rate)/100.0f);
    for (int i = 0; i < activations.size(); ++i) {
        if (dropout_mask[i] == 0) {
            activations[i] = 0;
        } else {
            activations[i] *= scale;
        }
    }
}

void DenseL::train_forward() {
    propagate();

    if (used_dropout) {
        dropout(dropout_rate);
    }

    switch(get_activation_func()) {
        case 0:
            reLU();
            break;
        case 1:
            softmax();
            break;
        default:
            std::cout << "invalid activation function";
            break;
    }
}

void DenseL::reset_gradients() {
    dropout_used_count.setZero();
    gradient_logits.setZero();
    gradient_sum_weights.setZero();
    gradient_sum_biases.setZero();
}