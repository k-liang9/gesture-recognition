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
    VectorXd dropout_mask{};
    VectorXd dropout_used_count{};
    MatrixXd gradient_logits{}; //this layer's neuron count * next layer's neuron count
    MatrixXd gradient_sum_weights{};
    VectorXd gradient_sum_biases{};

public:
    DenseL(MatrixXd *w, VectorXd *b, activation a); //todo: add next layer info, add new member variable declarations, add backprop variable resizings

    //forward propagation
    void propagate(MatrixXd& prev_activations);
    void dropout(float percent);
    void train_forward(); //todo: 2

    //backpropagation
    void calc_cost(VectorXd& expected); //stored in gradient_logits as they are parallel todo: CCEL
    void calc_gradient_activations(DenseL next_layer); //todo: 1
    void calc_gradients_weights(); //todo: 1
    void calc_gradients_biases(); //todo: 1
    void train_backward(); //todo: 2

    //normalization
    void reLU();
    void apply_reLU_derivative();
    void softmax();
    void calc_CCEL_derivative(VectorXd &expected);

    const MatrixXd& get_weights() const { return weights; }
    void set_weights(const MatrixXd& w) { weights = w; }
    const VectorXd& get_bias() const { return biases; }
    void set_bias(const VectorXd& b) { biases = b; }
    const VectorXd& get_dropout_mask() const { return dropout_mask; }
    void set_dropout_mask(const VectorXd& mask) { dropout_mask = mask; }
    const VectorXd& get_dropout_used_count() const { return dropout_used_count; }
    void set_dropout_used_count(const VectorXd& count) { dropout_used_count = count; }
    const MatrixXd& get_gradient_activations() const { return gradient_logits; }
    void set_gradient_activations(const MatrixXd& activations) { gradient_logits = activations; }
    const MatrixXd& get_gradient_sum_weights() const { return gradient_sum_weights; }
    void set_gradient_sum_weights(const MatrixXd& weights) { gradient_sum_weights = weights; }
    const VectorXd& get_gradient_sum_biases() const { return gradient_sum_biases; }
    void set_gradient_sum_biases(const VectorXd& biases) { gradient_sum_biases = biases; }
};


#endif //GESTURERECOGNITION_DENSEL_H
