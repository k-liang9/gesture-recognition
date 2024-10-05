#ifndef GESTURERECOGNITION_CONV_H
#define GESTURERECOGNITION_CONV_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "Layer.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Tensor;

class Conv : public Layer {
private:
    Tensor<double, 3> filter{};
    VectorXd biases{};
    Tensor<double, 3> feature_map;

public:
    Conv(Tensor<double, 3> *f, VectorXd* b, activation a);

    void flatten();
    void pool();
    void convolve(Tensor<double, 3> input);

public:
    const Tensor<double, 3>& get_filter() const { return filter; }
    void set_filter(const Tensor<double, 3>& f) { filter = f; }
    const MatrixXd& get_biases() const { return biases; }
    void set_biases(const MatrixXd& b) { biases = b; }
    const Tensor<double, 3>& get_activations() const { return feature_map; }
    void set_activations(const Tensor<double, 3>& a) { feature_map = a; }
};

#endif //GESTURERECOGNITION_CONV_H
