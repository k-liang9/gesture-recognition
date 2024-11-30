#include "DenseL.h"
#include "../Layer.h"
#include <Eigen/Dense>
#include <cassert>
#include "../../Inc/Global.h"

using namespace Eigen;

void DenseL::add_gradient_biases() {
    assert(gradient_sum_biases.size() == gradient_logits.size());
    gradient_sum_biases += gradient_logits;
}

void DenseL::add_gradient_weights(const VectorXd& prev_activation) {
    assert(gradient_sum_weights.cols() == get_activations().size());
    gradient_sum_weights += prev_activation.transpose() * gradient_logits;
}

void DenseL::calc_gradient_logits(const MatrixXd& next_weights, const VectorXd& next_gradient_logits) {
    assert(gradient_logits.size() == next_weights.cols());
    gradient_logits = (next_weights.transpose() * next_gradient_logits).rowwise().sum();
    apply_reLU_derivative();
}

//categorical cross-entropy loss
void DenseL::calc_CCEL_derivative(VectorXd &expected) {
    gradient_logits = get_activations() - expected;
}


void DenseL::backprop_nonoutput(const VectorXd& prev_activation,
                                const MatrixXd& next_weights, const VectorXd& next_gradient_logits) {
    calc_gradient_logits(next_weights, next_gradient_logits);
    add_gradient_weights(prev_activation);
    add_gradient_biases();
}

void DenseL::backprop_output(VectorXd& expected, const VectorXd& prev_activation) {
    calc_CCEL_derivative(expected);
    add_gradient_weights(prev_activation);
    add_gradient_biases();
}

void DenseL::change_params() {
    assert(weights.rows() == gradient_sum_weights.rows() && weights.cols() == gradient_sum_weights.cols());
    assert(biases.size() == gradient_sum_biases.size());

    MatrixXd gradient_weights = MatrixXd::Zero(weights.rows(), weights.cols());
    VectorXd gradient_biases = VectorXd::Zero(biases.size());
    if (used_dropout) {
        for (int i = 0; i < dropout_used_count.size(); ++i) {
            int cur_used_count = dropout_used_count[i];
            if (cur_used_count != 0) {
                gradient_weights.row(i) = gradient_sum_weights.row(i) / cur_used_count;
                gradient_biases[i] = gradient_sum_biases[i] / cur_used_count;
            }
        }
    } else {
        gradient_weights = gradient_sum_weights / Global::batch_size;
        gradient_biases = gradient_sum_biases / Global::batch_size;
    }

    biases -= gradient_biases;
    weights -= gradient_weights;

    dropout_used_count.setZero();
    gradient_logits.setZero();
    gradient_sum_weights.setZero();
    gradient_sum_biases.setZero();
}