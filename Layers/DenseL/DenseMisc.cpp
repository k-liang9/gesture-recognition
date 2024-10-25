#include "DenseL.h"
#include "../Layer.h"
#include <Eigen/Dense>

using namespace Eigen;

DenseL::DenseL(MatrixXd *w, VectorXd *b, activation a, bool dropout) :
        weights{*w},
        biases{*b},
        used_dropout{dropout} {
    set_activation_func(a);
    VectorXd activations = get_activations();
    activations.resize(biases.size());
    dropout_mask.resize(activations.size());
    dropout_used_count.resize(activations.size());
    gradient_sum_weights.resize(weights.rows(), weights.cols());
    gradient_sum_biases.resize(biases.size());
}

void DenseL::setup_neighbour_layers() {
    gradient_logits.resize(get_next_layer()->get_activations().size(), activations.size());
}