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
    int filter_rows = filter.dimension(0);
    int filter_cols = filter.dimension(1);
    int input_channels = filter.dimension(2);
    int num_filters = filter.dimension(3);

    for (int filter_index = 0; filter_index < num_filters; ++filter_index) { //per filter: this indexes the depth of this layer's fature map
        for (int channel = 0; channel < input_channels; ++channel) {
            for (int row = 0; row < feature_map.dimension(0) - filter_rows; ++row) {
                for (int col = 0; col < feature_map.dimension(1) - filter_cols; ++col) {
                    //todo: use convolve function instead of whatever's under here
                    double gradient = gradient_unpooled(row, col, filter_index);
                    for (int i = 0; i < filter_rows; ++i) {
                        for (int j = 0; j < filter_cols; ++j) {
                            gradient_sum_filter(i, j, channel, filter_index) +=
                                    gradient * feature_map(row+i, col+j, filter_index);
                        }
                    }
                }
            }
        }
    }

    //todo: set to 0 after changing params;
}

void ConvL::calc_gradient_feature_map() {
    gradient_feature_map.setZero();
}