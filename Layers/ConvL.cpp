#include "ConvL.h"
#include "Eigen/Dense"
#include <unsupported/Eigen/CXX11/Tensor>
#include <cassert>
#include <iostream>

using namespace Eigen;

ConvL::ConvL(Tensor<double, 4> *f, VectorXd* b, activation a) :
    filter(*f),
    biases(*b)
{
    set_activation_func(a);
}

void ConvL::flatten(const Tensor<double, 3>& input) {
    flat.resize(input.size());
    for (int i = 0; i < input.size(); ++i) {
        flat(i) = input.data()[i];
    }
}

//todo: implement more matrix operations
void ConvL::pool(const Tensor<double, 3>& input, const int pool_size) {
    pooled = Tensor<double, 3>(input.dimension(0)/pool_size, input.dimension(1)/pool_size, input.dimension(2));

    for (int channel = 0; channel < pooled.dimension(2); ++channel) {
        for (int pooled_row = 0; pooled_row < pooled.dimension(0); ++pooled_row) {
            for (int pooled_column = 0; pooled_column < pooled.dimension(1); ++pooled_column) {
                double max = std::numeric_limits<double>::min();
                for (int i = 0; i < pool_size; ++i) {
                    for (int j = 0; j < pool_size; ++j) {
                        int input_row = pooled_row * pool_size + i;
                        int input_col = pooled_column * pool_size + j;
                        if (input_row < input.dimension(0) && input_col < input.dimension(1)) {
                            double cur = input(input_row, input_col, channel);
                            if (cur > max) {
                                max = cur;
                            }
                        }
                    }
                }
                pooled(pooled_row, pooled_column, channel) = max;
            }
        }
    }
}

//todo: implement more matrix operations
void ConvL::convolve(const Tensor<double, 3>& input) {
    assert(input.dimension(2) == filter.dimension(2));

    feature_map.resize(input.dimension(0) - filter.dimension(0) + 1,
                input.dimension(1) - filter.dimension(1) + 1,
                filter.dimension(3));

    //memory usage tradeoff: calculate feature map by feature map instead of calculating every feature map at once
    for (int filter_index = 0; filter_index < filter.dimension(3); ++filter_index) {
        for (int i = 0; i < input.dimension(0); ++i) {
            for (int j = 0; j < input.dimension(1); ++j) {
                //create patches
                Eigen::Tensor<double, 3> filter_slice = filter.chip(filter_index, 3);
                Eigen::Tensor<double, 3> input_slice = input.slice(Eigen::array<long, 3>{i, j, 0},
                                                                   filter_slice.dimensions());

                //sum resulting patch
                Tensor<double, 3> result = (filter_slice * input_slice).sum();
                for (int k = 0; k < result.dimension(0); ++k) {
                    for (int l = 0; l < result.dimension(1); ++l) {
                        for (int m = 0; m < result.dimension(2); ++m) {
                            feature_map(i, j, filter_index) += result(k, l, m);
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < feature_map.dimension(2); ++i) {
        feature_map.chip(i, 2) = feature_map.chip(i, 2) - biases(i);
    }
}