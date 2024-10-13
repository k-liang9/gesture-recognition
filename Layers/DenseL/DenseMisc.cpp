#include "DenseL.h"
#include "../Layer.h"
#include <Eigen/Dense>
#include <cassert>
#include <random>

using namespace Eigen;

DenseL::DenseL(MatrixXd *w, VectorXd *b, activation a) :
        weights{*w},
        biases{*b} {
    set_activation_func(a);
    dropout_mask.resize(biases.size());
    activations.resize(biases.size());
    dropout_used_count.resize(biases.size());
    assert(weights.rows() == weights.size());
}