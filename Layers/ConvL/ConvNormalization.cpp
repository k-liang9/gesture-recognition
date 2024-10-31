#include "ConvL.h"
#include "Eigen/Dense"
#include <unsupported/Eigen/CXX11/Tensor>
#include <cassert>
#include <iostream>

void ConvL::reLU() {
    for (int i = 0; i < feature_map.size(); ++i) {
        if (feature_map.data()[i] < 0) {
            feature_map.data()[i] = 0;
        }
    }
}

void ConvL::apply_reLU_derivative() {
    for (int i = 0; i < gradient_pooled.size(); ++i) {
        double& cur = gradient_pooled.data()[i];
        if (cur > 0) {
            cur *= 1;
        } else {
            cur = 0;
        }
    }
}