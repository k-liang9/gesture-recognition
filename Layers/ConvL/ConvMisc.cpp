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
    //todo: resize variables
}