#ifndef GESTURERECOGNITION_DENSE_H
#define GESTURERECOGNITION_DENSE_H
#include <Eigen/Dense>
#include <utility>
#include "Layer.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

class Dense : public Layer {
private:
    MatrixXd weights{};
    VectorXd biases{};
    VectorXd activations{};
    VectorXd dropout_mask{};

public:
    Dense(MatrixXd *w, VectorXd *b, activation a);

    void propagate(MatrixXd& prev_activations);

    void dropout(float percent);

    const MatrixXd& get_weights() const { return weights; }
    void set_weights(const MatrixXd& w) { weights = w; }
    const VectorXd& get_bias() const { return biases; }
    void set_bias(const VectorXd& b) { biases = b; }
    const VectorXd& get_dropout_mask() const { return dropout_mask; }
    void set_dropout_mask(const VectorXd& mask) { dropout_mask = mask; }
};


#endif //GESTURERECOGNITION_DENSE_H
