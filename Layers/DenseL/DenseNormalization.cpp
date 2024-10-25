#include "DenseL.h"
#include "../Layer.h"
#include <Eigen/Dense>
#include <cassert>
#include <random>

using namespace Eigen;

void DenseL::reLU() {
    VectorXd& activations = get_activations();
    for (int i = 0; i < activations.size(); ++i) {
        if (activations[i] < 0) {
            activations[i] = 0;
        }
    }
}

void DenseL::softmax() {
    VectorXd& activations = get_activations();
    activations = activations.unaryExpr([](double x) {return std::exp(x);});
    double sum = activations.sum();
    activations = activations.unaryExpr([sum](double x) {return x/sum;});
}

void DenseL::apply_reLU_derivative() {
    VectorXd& activations = get_activations();
    assert(activations.size() == gradient_logits.rows());
    for (int i = 0; i < activations.size(); ++i) {
        if (activations[i] > 0) {
            gradient_logits.row(i) *= 1;
        } else {
            gradient_logits.row(i).setZero();
        }
    }
}