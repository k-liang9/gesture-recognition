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