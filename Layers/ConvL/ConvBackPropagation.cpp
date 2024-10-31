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

    gradient_pooled = next_layer->get_gradient_feature_map();
}

void ConvL::unpool() {
    int rows = gradient_pooled.dimension(0);
    int cols = gradient_pooled.dimension(1);
    int channels  = gradient_pooled.dimension(2);

    gradient_feature_map.resize(rows*pool_size, cols*pool_size, channels);
    gradient_feature_map.setZero();

    for (int channel = 0; channel < channels; ++channel) {
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                std::pair<int, int>& cur_index = pooled_index(row, col, channel);
                gradient_feature_map(row*pool_size + cur_index.first,
                                     col*pool_size + cur_index.second, channel) =
                                             gradient_pooled(row, col, channel);
            }
        }
    }
}