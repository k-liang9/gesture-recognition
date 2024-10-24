#ifndef GESTURERECOGNITION_LAYER_H
#define GESTURERECOGNITION_LAYER_H
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

enum activation {
    reLU,
    softmax
};

class Layer {
private:
    activation activ_func{};
    VectorXd activations{};

public:
    static void reLU(Tensor<double, 3>& input);

    static void extract_layer(Tensor<double, 3> &tensor, MatrixXd &matrix, int i);

    const activation& get_activation_func() const { return activ_func; }
    void set_activation_func(const activation& a) { activ_func = a; }
    const VectorXd& get_activations() const { return activations; }
    void set_activations(const VectorXd& a) { activations = a; }
};


#endif //GESTURERECOGNITION_LAYER_H
