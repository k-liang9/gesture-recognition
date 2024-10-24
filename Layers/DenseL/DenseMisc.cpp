#include "DenseL.h"
#include "../Layer.h"
#include <Eigen/Dense>
#include <cassert>
#include <random>

using namespace Eigen;

DenseL::DenseL(MatrixXd *w, VectorXd *b, activation a, bool dropout) :
        weights{*w},
        biases{*b},
        used_dropout{dropout} {
    set_activation_func(a);
    activations.resize(biases.size());
    dropout_mask.resize(activations.size());
    dropout_used_count.resize(activations.size());
    gradient_sum_weights.resize(weights.rows(), weights.cols());
    gradient_sum_biases.resize(biases.size());
}

void DenseL::link_layers(DenseL *prev_layer, DenseL *next_layer) {
    set_next_layer(next_layer);
    set_prev_layer(prev_layer);
    gradient_logits.resize(next_layer->get_activations().size(), activations.size());
}