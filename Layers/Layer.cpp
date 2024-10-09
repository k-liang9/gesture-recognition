#include "Layer.h"

using namespace Eigen;

void Layer::extract_layer(Tensor<double, 3> &tensor, MatrixXd &matrix, int i) {
    Eigen::Tensor<double, 2> slice = tensor.chip(i, 2);
    Eigen::Map<Eigen::MatrixXd> mappedMatrix(slice.data(), slice.dimension(0), slice.dimension(1));
    matrix = mappedMatrix;(slice.data(), slice.dimension(0), slice.dimension(1));
}


void Layer::reLU(Tensor<double, 3>& input) {
    for (int i = 0; i < input.size(); ++i) {
        if (input.data()[i] < 0) {
            input.data()[i] = 0;
        }
    }
}

void Layer::reLU(VectorXd& input) {
    for (int i = 0; i < input.size(); ++i) {
        if (input.data()[i] < 0) {
            input.data()[i] = 0;
        }
    }
}

void Layer::softmax(VectorXd& input) {
    input = input.unaryExpr([](double x) {return std::exp(x);});
    double sum = input.sum();
    input = input.unaryExpr([sum](double x) {return x/sum;});
}

void Layer::do_activation(Layer& layer, auto& pre_activation) {
    switch (layer.get_activation_func()) {
        case 0: //reLU
            reLU(pre_activation);
            break;
        case 1: //softmax
            softmax(pre_activation);
            break;
    }
}