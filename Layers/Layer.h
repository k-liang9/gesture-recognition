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

public:
    static void reLU(Tensor<double, 3>& input);
    static void reLU(VectorXd& input);

    static void softmax(VectorXd& input);

    static void extract_layer(Tensor<double, 3> &tensor, MatrixXd &matrix, int i);

    static void do_activation(Layer& layer, auto& pre_activation);

    const activation& get_activation_func() const { return activ_func; }
    void set_activation_func(const activation& a) { activ_func = a; }
};


#endif //GESTURERECOGNITION_LAYER_H
