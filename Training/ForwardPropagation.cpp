#include "ForwardPropagation.h"
#include "../Layer.h"
#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

void calculateActivations(const VectorXd &prev_activations,
                          VectorXd &cur_activation,
                          const MatrixXd &weights,
                          const std::function<VectorXd(VectorXd)> &activation_function,
                          const VectorXd& bias) {
    assert(prev_activations.size() == weights.rows());
    cur_activation = weights * prev_activations;
    
    assert(cur_activation.size() == bias.size());
    cur_activation -= bias;
    
    cur_activation = activation_function(cur_activation);
}

VectorXd tanh(const VectorXd &x) {
    return x.unaryExpr([](double x) { return std::tanh(x); });
}

VectorXd sigmoid(const VectorXd &x) {
    return x.unaryExpr([](double x) { return 1 / (1 + std::exp(-x)); });
}