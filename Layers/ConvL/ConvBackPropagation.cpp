#include "ConvL.h"
#include "Eigen/Dense"
#include <unsupported/Eigen/CXX11/Tensor>
#include <cassert>
#include <iostream>

void ConvL::unflatten() {
    const int rows = pooled.dimension(0);
    const int cols = pooled.dimension(1);
    const int channels = pooled.dimension(2);

    pooled_gradient.resize(rows, cols, channels);

    int index = 0;

}