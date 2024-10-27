#ifndef GESTURERECOGNITION_CONVL_H
#define GESTURERECOGNITION_CONVL_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../Layer.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Tensor;

class ConvL : public Layer {
private:
    Tensor<double, 4> filter{};
    VectorXd biases{};
    Tensor<double, 3> feature_map{};
    const int pool_size{2};
    Tensor<double, 3> pooled{};
    Tensor<std::pair<int, int>, 3> pooled_index{};
    Tensor<double, 3> pooled_gradient{};

public:
    //misc
    ConvL(Tensor<double, 4> *f, VectorXd* b, activation a);
    void setup_neighbour_layers(); //todo: is this needed?

    //forward propagation
    void flatten(const Tensor<double, 3>& input);
    void pool(const Tensor<double, 3>& input);
    void convolve(const Tensor<double, 3>& input);

    //backprop
    void unflatten();

    //normalization
    void reLU();
    void apply_reLU_derivative();

    //getters
    const Tensor<double, 4>& get_filter() const { return filter; }
    const VectorXd& get_biases() const { return biases; }
    const Tensor<double, 3>& get_feature_map() const { return feature_map; }

    //setters
    void set_filter(const Tensor<double, 4>& f) { filter = f; }
    void set_biases(const VectorXd& b) { biases = b; }
    void set_feature_map(const Tensor<double, 3>& a) { feature_map = a; }
};

#endif //GESTURERECOGNITION_CONVL_H
