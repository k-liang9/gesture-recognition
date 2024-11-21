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
    Tensor<double, 3> gradient_pooled{}; //next layer
    Tensor<double, 3> gradient_unpooled{}; //next layer

    Tensor<double, 4> gradient_sum_filter{};
    VectorXd gradient_sum_biases{};
    Tensor<double, 3> gradient_feature_map{};
    bool is_last{};

public:
    //misc
    ConvL(Tensor<double, 4> *f, VectorXd* b, activation a, bool last);
    void setup_neighbour_layers(); //todo: is this needed?
    void convolve(const Tensor<double, 2>& input, const Tensor<double, 2>& kernel,
                  Tensor<double, 2>& output);
    void convolve_full(const Tensor<double, 2>& input, const Tensor<double, 2>& kernel,
                       Tensor<double, 2>& output);

    //forward propagation
    void flatten();
    void pool();
    void apply_filter(const Tensor<double, 3>& input); //todo: link
    void train_forward();

    //backprop
    void unflatten();
    void copy_next_gradient();
    void unpool();
    void calc_gradient_feature_map(); //full convolution(F, 180˚ rotated loss gradient)
    void add_gradient_filters();
    void add_gradient_biases();
    void change_params();
    void train_backward();
    //unflatten/copy gradient, unrelu, unpool

    //normalization
    void reLU();
    void apply_reLU_derivative();

    //getters
    const Tensor<double, 3>& get_pooled() const { return pooled; }
    const Tensor<std::pair<int, int>, 3>& get_pooled_index() const { return pooled_index; }
    const Tensor<double, 3>& get_gradient_pooled() const { return gradient_pooled; }
    const Tensor<double, 4>& get_gradient_sum_filter() const { return gradient_sum_filter; }
    const VectorXd& get_gradient_sum_biases() const { return gradient_sum_biases; }
    const Tensor<double, 3>& get_gradient_unpooled() const { return gradient_unpooled; }
    const bool get_is_last() const { return is_last; }

    //setters
    void set_pooled(const Tensor<double, 3>& p) { pooled = p; }
    void set_pooled_index(const Tensor<std::pair<int, int>, 3>& pi) { pooled_index = pi; }
    void set_gradient_pooled(const Tensor<double, 3>& gp) { gradient_pooled = gp; }
    void set_gradient_sum_filter(const Tensor<double, 4>& gsf) { gradient_sum_filter = gsf; }
    void set_gradient_sum_biases(const VectorXd& gsb) { gradient_sum_biases = gsb; }
    void set_gradient_unpooled(const Tensor<double, 3>& gfm) { gradient_unpooled = gfm; }
    const bool set_is_last(const bool last) { is_last = last; }
};

#endif //GESTURERECOGNITION_CONVL_H