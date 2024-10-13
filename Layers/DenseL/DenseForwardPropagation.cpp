#include "DenseL.h"
#include "../Layer.h"
#include <Eigen/Dense>
#include <cassert>
#include <random>

using namespace Eigen;

void DenseL::propagate(MatrixXd& prev_activations) {
    assert(prev_activations.size() == weights.cols());
    activations = weights*prev_activations-biases;
}

void DenseL::dropout(float dropout_rate) {
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