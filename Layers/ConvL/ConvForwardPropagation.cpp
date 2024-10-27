#include "ConvL.h"
#include "Eigen/Dense"
#include <unsupported/Eigen/CXX11/Tensor>
#include <cassert>
#include <iostream>

using namespace Eigen;

void ConvL::flatten(const Tensor<double, 3>& input) {
    VectorXd& activations = get_activations();
    activations.resize(input.size());
    for (int i = 0; i < input.size(); ++i) {
        activations(i) = input.data()[i];
    }
}

void ConvL::pool(const Tensor<double, 3>& input) {
    int pooled_rows = input.dimension(0) / pool_size;
    int pooled_cols = input.dimension(1) / pool_size;
    int channels = input.dimension(2);

    pooled.resize(pooled_rows, pooled_cols, channels);
    pooled_index.resize(pooled_rows, pooled_cols, channels);

    for (int channel = 0; channel < channels; ++channel) {
        for (int pooled_row = 0; pooled_row < pooled_rows; ++pooled_row) {
            for (int pooled_col = 0; pooled_col < pooled_cols; ++pooled_col) {
                double max_val = std::numeric_limits<double>::lowest();
                std::pair<int, int> max_index{};
                for (int i = 0; i < pool_size; ++i) {
                    for (int j = 0; j < pool_size; ++j) {
                        int input_row = pooled_row * pool_size + i;
                        int input_col = pooled_col * pool_size + j;
                        if (input_row < input.dimension(0) && input_col < input.dimension(1)) {
                            double cur_val = input(input_row, input_col, channel);
                            if (cur_val > max_val) {
                                max_val = cur_val;
                                max_index = {i, j};
                            }
                        }
                    }
                }
                pooled(pooled_row, pooled_col, channel) = max_val;
                pooled_index(pooled_row, pooled_col, channel) = max_index;
            }
        }
    }
}


void sum_patch(Tensor<double, 3>& result, Tensor<double, 3>& feature_map, int row, int col, int filter_index) {
    for (int i = 0; i < result.dimension(0); ++i) {
        for (int j = 0; j < result.dimension(1); ++j) {
            for (int k = 0; k < result.dimension(2); ++k) {
                feature_map(row, col, filter_index) += result(i, j, k);
            }
        }
    }
}

void ConvL::convolve(const Tensor<double, 3>& input) {
    assert(input.dimension(2) == filter.dimension(2));

    int feature_map_rows = input.dimension(0) - filter.dimension(0) + 1;
    int feature_map_cols = input.dimension(1) - filter.dimension(1) + 1;
    int num_filters = filter.dimension(3);

    feature_map.resize(feature_map_rows, feature_map_cols, num_filters);

    //memory usage tradeoff: calculate feature map by feature map instead of calculating every feature map at once
    for (int filter_index = 0; filter_index < num_filters; ++filter_index) {
        for (int i = 0; i < feature_map_rows; ++i) {
            for (int j = 0; j < feature_map_cols; ++j) {
                //create patches
                Eigen::Tensor<double, 3> filter_slice = filter.chip(filter_index, 3);
                Eigen::Tensor<double, 3> input_slice = input.slice(Eigen::array<long, 3>{i, j, 0}, filter_slice.dimensions());

                //sum resulting patch
                Tensor<double, 3> result = (filter_slice * input_slice).sum();
                sum_patch(result, feature_map, i, j, filter_index);
            }
        }
    }

    for (int i = 0; i < feature_map.dimension(2); ++i) {
        feature_map.chip(i, 2) = feature_map.chip(i, 2) - biases(i);
    }
}