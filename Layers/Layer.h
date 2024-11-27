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
    Layer() {}
    virtual ~Layer() = default;
    static void extract_layer(Tensor<double, 3> &tensor, MatrixXd &matrix, int i);

    //getters
    activation& get_activation_func() { return activ_func; }
    VectorXd& get_activations() { return activations; }

    //setters
    void set_activation_func(const activation& a) { activ_func = a; }
};


#endif //GESTURERECOGNITION_LAYER_H
