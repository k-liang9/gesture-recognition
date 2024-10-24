#include "DenseL.h"
#include "../Layer.h"
#include <Eigen/Dense>
#include <cassert>
#include "../../Global.h"

using namespace Eigen;

void DenseL::add_gradient_biases() {
    assert(gradient_sum_biases.size() == gradient_logits.rows());
    gradient_sum_biases += gradient_logits.rowwise().sum();
}

void DenseL::add_gradient_weights() {
    assert(gradient_sum_weights.rows() == gradient_logits.rows() && gradient_sum_weights.cols() == activations.size());
    gradient_sum_weights += gradient_logits * prev_layer->get_activations().transpose();
}

void DenseL::calc_gradient_logits() {
    assert(gradient_logits.rows() == next_layer->get_weights().cols() && gradient_logits.cols() == next_layer->get_gradient_logits().cols());
    gradient_logits.resize(next_layer->get_weights().cols(), next_layer->get_gradient_logits().cols());
    gradient_logits = next_layer->get_weights().transpose() * next_layer->get_gradient_logits();
    apply_reLU_derivative();
}

//categorical cross-entropy loss
void DenseL::calc_CCEL_derivative(VectorXd &expected) {
    gradient_logits.col(0) = activations - expected;
}


void DenseL::backprop_nonoutput() {
    calc_gradient_logits();
    add_gradient_weights();
    add_gradient_biases();
}

void DenseL::backprop_output(VectorXd& expected) {
    calc_CCEL_derivative(expected);
    add_gradient_weights();
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
}