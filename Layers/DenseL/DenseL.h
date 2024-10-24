#ifndef GESTURERECOGNITION_DENSEL_H
#define GESTURERECOGNITION_DENSEL_H
#include <Eigen/Dense>
#include <utility>
#include "../Layer.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

class DenseL : public Layer {
private:
    MatrixXd weights{};
    VectorXd biases{};
    VectorXd activations{};
    VectorXi dropout_mask{};
    VectorXi dropout_used_count{};
    MatrixXd gradient_logits{}; //this layer's neuron count * next layer's neuron count
    MatrixXd gradient_sum_weights{};
    VectorXd gradient_sum_biases{};
    bool used_dropout;
    DenseL* next_layer;
    Layer* prev_layer;

public:
    DenseL(MatrixXd *w, VectorXd *b, activation a, bool dropout);

    void link_layers(DenseL* prev_layer, DenseL* next_layer);

    //forward propagation
    void propagate(MatrixXd& prev_activations);
    void dropout(float percent);
    void train_forward(); //todo: 2

    //backpropagation
    void calc_loss(VectorXd& expected); //CCEL //stored in gradient_logits as they are parallel
    void calc_gradient_logits(DenseL next_layer);
    void calc_gradients_weights();
    void calc_gradients_biases();
    void train_backward(); //todo: 2
    void change_params();

    //normalization
    void reLU();
    void apply_reLU_derivative();
    void softmax();
    void calc_CCEL_derivative(VectorXd &expected);

    const MatrixXd& get_weights() const { return weights; }
    const VectorXd& get_biases() const { return biases; }
    const VectorXd& get_activations() const { return activations; }
    const VectorXi& get_dropout_mask() const { return dropout_mask; }
    const VectorXi& get_dropout_used_count() const { return dropout_used_count; }
    const MatrixXd& get_gradient_logits() const { return gradient_logits; }
    const MatrixXd& get_gradient_sum_weights() const { return gradient_sum_weights; }
    const VectorXd& get_gradient_sum_biases() const { return gradient_sum_biases; }
    bool get_used_dropout() const { return used_dropout; }
    DenseL* get_next_layer() const { return next_layer; }
    Layer* get_prev_layer() const { return prev_layer; }

    // Setters
    void set_weights(const MatrixXd& w) {}
    void set_biases(const VectorXd& b) { biases = b; }
    void set_activations(const VectorXd& a) { activations = a; }
    void set_dropout_mask(const VectorXi& dm) { dropout_mask = dm; }
    void set_dropout_used_count(const VectorXi& duc) { dropout_used_count = duc; }
    void set_gradient_logits(const MatrixXd& gl) { gradient_logits = gl; }
    void set_gradient_sum_weights(const MatrixXd& gsw) { gradient_sum_weights = gsw; }
    void set_gradient_sum_biases(const VectorXd& gsb) { gradient_sum_biases = gsb; }
    void set_used_dropout(bool ud) { used_dropout = ud; }
    void set_next_layer(DenseL* nl) { next_layer = nl; }
    void set_prev_layer(DenseL* pl) { prev_layer = pl; }
};


#endif //GESTURERECOGNITION_DENSEL_H
