#include "ConvL.h"
#include "Eigen/Dense"
#include <unsupported/Eigen/CXX11/Tensor>
#include <cassert>
#include <iostream>
#include "../DenseL/DenseL.h"

void ConvL::unflatten() {
    const int rows = pooled.dimension(0);
    const int cols = pooled.dimension(1);
    const int channels = pooled.dimension(2);

    gradient_pooled.resize(rows, cols, channels);

    DenseL* next_layer{dynamic_cast<DenseL*>(get_next_layer())};
    if (!next_layer) {
        throw std::runtime_error("Next layer is not of type DenseL");
    }
    VectorXd gradient_logits = next_layer->get_gradient_logits();

    assert(rows*cols*channels == gradient_logits.size());

    int index = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int k = 0; k < channels; ++k) {
                gradient_pooled(i, j, k) = gradient_logits(index);
                index++;
            }
        }
    }
}

void ConvL::copy_next_gradient() {
    ConvL* next_layer{dynamic_cast<ConvL*>(get_next_layer())};
    if (!next_layer) {
        throw std::runtime_error("Next layer is not of type ConvL");
    }

    gradient_pooled = next_layer->get_gradient_unpooled();
}

void ConvL::unpool() {
    int rows = gradient_pooled.dimension(0);
    int cols = gradient_pooled.dimension(1);
    int channels  = gradient_pooled.dimension(2);

    gradient_unpooled.resize(rows * pool_size, cols * pool_size, channels);
    gradient_unpooled.setZero();

    for (int channel = 0; channel < channels; ++channel) {
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                std::pair<int, int>& cur_index = pooled_index(row, col, channel);
                gradient_unpooled(row * pool_size + cur_index.first,
                                     col*pool_size + cur_index.second, channel) =
                                             gradient_pooled(row, col, channel);
            }
        }
    }
}

void ConvL::add_gradient_filters() {
    int input_channels = filter.dimension(2);
    int num_filters = filter.dimension(3);

    assert(input_channels == feature_map.dimension(2));
    assert(num_filters == gradient_unpooled.dimension(2));

    for (int filter_index = 0; filter_index < num_filters; ++filter_index) { //per filter: this indexes the depth of this layer's feature map

        Tensor<double, 2> kernel_slice = gradient_unpooled.chip(filter_index, 2);
        for (int channel = 0; channel < input_channels; ++channel) {
            Tensor<double, 2> input_slice = feature_map.chip(channel, 2);
            Tensor<double, 2> output_slice = gradient_sum_filter
                    .chip(filter_index, 3).chip(channel, 2);

            convolve(input_slice, kernel_slice, output_slice);
        }
    }
}

//for each output channel, convolve each channel of the 3d filter tensor, sum each slice, average
void ConvL::calc_gradient_feature_map() {
    int input_channels = feature_map.dimension(2);
    int output_channels = gradient_unpooled.dimension(2);

    assert(input_channels == filter.dimension(2));
    assert(output_channels == filter.dimension(3));

    gradient_feature_map.reshape(feature_map.dimensions());
    gradient_feature_map.setZero();

    for (int input_channel = 0; input_channel < input_channels; ++input_channel) {

        Tensor<double, 2> input_slice = gradient_feature_map.chip(input_channel, 2);
        for (int output_channel = 0; output_channel < output_channels; ++output_channel) {
            Tensor<double, 2> output_slice = gradient_unpooled.chip(output_channel, 2).
                    reverse(Eigen::array<int, 2>({1, 0}));
            Tensor<double, 2> filter_slice = filter.chip(output_channel, 3).chip(input_channel, 2);
            convolve_full(filter_slice, output_slice, input_slice);
        }

        input_slice = input_slice / static_cast<double>(output_channels);
    }
}