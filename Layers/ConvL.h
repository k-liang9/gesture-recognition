#ifndef GESTURERECOGNITION_CONVL_H
#define GESTURERECOGNITION_CONVL_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "Layer.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Tensor;

class ConvL : public Layer {
private:
    Tensor<double, 4> filter{};
    VectorXd biases{};
    Tensor<double, 3> feature_map{};
    VectorXd flat{};
    Tensor<double, 3> pooled{};

public:
    ConvL(Tensor<double, 4> *f, VectorXd* b, activation a);

    void flatten(const Tensor<double, 3>& input);
    void pool(const Tensor<double, 3>& input, const int pool_size);
    void convolve(const Tensor<double, 3>& input);

public:
    const Tensor<double, 4>& get_filter() const { return filter; }
    void set_filter(const Tensor<double, 4>& f) { filter = f; }
    const VectorXd& get_biases() const { return biases; }
    void set_biases(const VectorXd& b) { biases = b; }
    const Tensor<double, 3>& get_feature_map() const { return feature_map; }
    void set_feature_map(const Tensor<double, 3>& a) { feature_map = a; }
};

#endif //GESTURERECOGNITION_CONVL_H
