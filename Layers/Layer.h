#ifndef GESTURERECOGNITION_LAYER_H
#define GESTURERECOGNITION_LAYER_H
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::MatrixXd;
using Eigen::Tensor;
using Eigen::VectorXd;

enum activation {
    reLU,
    softmax
};

class Layer {
private:
    activation activ_func{};

public:
    template <typename T>
    static void reLU(T&);

    template <typename T>
    static void softmax(T&);

    static void extract_layer(Tensor<double, 3> &tensor, MatrixXd &matrix, int i);

    const activation& get_activation_func() const { return activ_func; }
    void set_activation_func(const activation& a) { activ_func = a; }
};


#endif //GESTURERECOGNITION_LAYER_H
