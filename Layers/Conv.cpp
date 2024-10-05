#include "Conv.h"
#include "Eigen/Dense"
#include <unsupported/Eigen/CXX11/Tensor>
#include <cassert>

using namespace Eigen;

Conv::Conv(Tensor<double, 3> *f, VectorXd* b, activation a) :
    filter(*f),
    biases(*b)
{
    set_activation_func(a);
}

void Conv::flatten() {

}

void Conv::pool() {

}

void Conv::convolve(Tensor<double, 3> input) {
    feature_map(input.dimension(0) - filter.dimension(0) + 1,
                input.dimension(1) - filter.dimension(1) + 1,
                filter.dimension(2));

    //memory usage tradeoff: calculate feature map by feature map instead of calculating every feature map at once
    for (int filter_index = 0; filter_index < filter.dimension(2); ++filter_index) {
        for (int input_layer = 0; input_layer < input.dimension(2); ++input_layer) {
            array<IndexPair<double>, 1> convolution_dims = {IndexPair<double>(0, 0)};
            feature_map.chip(filter_index, 2) += input.chip(input_layer, 2)
                    .contract(filter.chip(filter_index, 2), convolution_dims)
                    .sum();
        }
    }

    for (int i = 0; i < feature_map.dimension(2); ++i) {
        feature_map.chip(i, 2) = feature_map.chip(i, 2) - biases(i);
    }
}