#include "DenseL.h"
#include "../Layer.h"
#include <Eigen/Dense>
#include <cassert>
#include <random>

using namespace Eigen;

void DenseL::calc_cost(VectorXd& expected) {
    assert(gradient_logits.cols() == 1);
    assert(gradient_logits.rows() == expected.size());

    gradient_logits = 2 * (activations.col(0) - expected);
    gradient_logits = (gradient_logits.transpose() *
                       calc_activation_derivative()).transpose();
}