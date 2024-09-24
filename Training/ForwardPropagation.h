#ifndef GESTURERECOGNITION_FORWARDPROPAGATION_H
#define GESTURERECOGNITION_FORWARDPROPAGATION_H
#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;
void calculateActivations(const VectorXd &prev_activations, VectorXd &cur_activation, const MatrixXd &weights);

void tanh(VectorXd& pre_activation);

void sigmoid(VectorXd& pre_activation);

#endif //GESTURERECOGNITION_FORWARDPROPAGATION_H
