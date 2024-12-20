#include "Layer.h"

using namespace Eigen;

void Layer::extract_layer(Tensor<double, 3> &tensor, MatrixXd &matrix, int i) {
    Eigen::Tensor<double, 2> slice = tensor.chip(i, 2);
    Eigen::Map<Eigen::MatrixXd> mappedMatrix(slice.data(), slice.dimension(0), slice.dimension(1));
    matrix = mappedMatrix;(slice.data(), slice.dimension(0), slice.dimension(1));
}