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
    VectorXi dropout_mask{};
    VectorXi dropout_used_count{};
    VectorXd gradient_logits{};
    MatrixXd gradient_sum_weights{};
    VectorXd gradient_sum_biases{};
    bool used_dropout{};
    float dropout_rate{};

public:
    DenseL(MatrixXd *w, VectorXd *b, activation a, bool dropout);
    void setup_neighbour_layers(); //todo: is this needed?

    //forward propagation
    void propagate();
    void dropout(float percent);
    void train_forward();

    //backpropagation
    void calc_loss(VectorXd& expected); //CCEL //stored in gradient_logits as they are parallel
    void calc_gradient_logits();
    void calc_CCEL_derivative(VectorXd &expected);
    void add_gradient_weights();
    void add_gradient_biases();
    void backprop_nonoutput();
    void backprop_output(VectorXd& expected);
    void change_params();
    void reset_gradients();

    //normalization
    void reLU();
    void apply_reLU_derivative();
    void softmax();

    //getters
    const MatrixXd& get_weights() const { return weights; }
    const VectorXd& get_biases() const { return biases; }
    const VectorXi& get_dropout_mask() const { return dropout_mask; }
    const VectorXi& get_dropout_used_count() const { return dropout_used_count; }
    const VectorXd& get_gradient_logits() const { return gradient_logits; }
    const MatrixXd& get_gradient_sum_weights() const { return gradient_sum_weights; }
    const VectorXd& get_gradient_sum_biases() const { return gradient_sum_biases; }
    bool get_used_dropout() const { return used_dropout; }
    float get_dropout_rate() const { return dropout_rate; }

    //setters
    void set_weights(const MatrixXd& w) {}
    void set_biases(const VectorXd& b) { biases = b; }
    void set_dropout_mask(const VectorXi& dm) { dropout_mask = dm; }
    void set_dropout_used_count(const VectorXi& duc) { dropout_used_count = duc; }
    void set_gradient_logits(const VectorXd& gl) { gradient_logits = gl; }
    void set_gradient_sum_weights(const MatrixXd& gsw) { gradient_sum_weights = gsw; }
    void set_gradient_sum_biases(const VectorXd& gsb) { gradient_sum_biases = gsb; }
    void set_used_dropout(bool ud) { used_dropout = ud; }
    void set_dropout_rate(float dr) { dropout_rate = dr; }
};


#endif //GESTURERECOGNITION_DENSEL_H
