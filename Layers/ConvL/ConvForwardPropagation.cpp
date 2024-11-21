#include "ConvL.h"
#include "Eigen/Dense"
#include <unsupported/Eigen/CXX11/Tensor>
#include <cassert>
#include <iostream>

using namespace Eigen;

void ConvL::flatten() {
    VectorXd& activations = get_activations();
    activations.resize(pooled.size());
    for (int i = 0; i < pooled.size(); ++i) {
        activations(i) = pooled.data()[i];
    }
}

void ConvL::pool() {
    int pooled_rows = feature_map.dimension(0) / pool_size;
    int pooled_cols = feature_map.dimension(1) / pool_size;
    int channels = feature_map.dimension(2);

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
                        if (input_row < feature_map.dimension(0) && input_col < feature_map.dimension(1)) {
                            double cur_val = feature_map(input_row, input_col, channel);
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

void ConvL::apply_filter(const Tensor<double, 3>& input) {
    int feature_map_rows = input.dimension(0) - filter.dimension(0) + 1;
    int feature_map_cols = input.dimension(1) - filter.dimension(1) + 1;
    int num_filters = filter.dimension(3);
    int num_channels = filter.dimension(2);

    assert(input.dimension(2) == filter.dimension(2));

    feature_map.resize(feature_map_rows, feature_map_cols, num_filters);

    //memory usage tradeoff: calculate feature map by feature map instead of calculating every feature map at once
    for (int filter_index = 0; filter_index < num_filters; ++filter_index) {

        Tensor<double, 2> output_slice = feature_map.chip(filter_index, 2);
        for (int channel = 0; channel < num_channels; ++channel) {
            Tensor<double, 2> input_slice = input.chip(channel, 2);
            Tensor<double, 2> filter_slice = filter.chip(filter_index, 3).chip(channel, 2);

            convolve(input_slice, filter_slice, output_slice);
        }

        output_slice = output_slice / static_cast<double>(num_filters) - biases(filter_index);
    }
}

void ConvL::train_forward() {
    apply_filter();
    pool();
    if (is_last) {
        flatten();
    }
}