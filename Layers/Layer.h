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
    Layer* next_layer{};
    Layer* prev_layer{};

public:
    Layer() {}
    virtual ~Layer() = default;
    static void extract_layer(Tensor<double, 3> &tensor, MatrixXd &matrix, int i);
    void link_layers(Layer* prev_layer, Layer* next_layer);
    void setup_neighbour_layers();

    //getters
    activation& get_activation_func() { return activ_func; }
    VectorXd& get_activations() { return activations; }
    Layer* get_next_layer() { return next_layer; }
    Layer* get_prev_layer() { return prev_layer; }

    //setters
    void set_activation_func(const activation& a) { activ_func = a; }
    void set_next_layer(Layer* nl) { next_layer = nl; }
    void set_prev_layer(Layer* pl) { prev_layer = pl; }
};


#endif //GESTURERECOGNITION_LAYER_H
