#include "Dense.h"
#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;
Dense::Dense(int layer_index, VectorXd bias, MatrixXd weights, int neuron_count, int prev_layer_neuron_count = 0, int next_layer_neuron_count = 0) :
        m_layer_index{layer_index},
        m_neuron_count{neuron_count},
        m_prev_layer_neuron_count{prev_layer_neuron_count},
        m_next_layer_neuron_count{next_layer_neuron_count},
        m_bias{std::move(bias)},
        m_weights{std::move(weights)}
{
    m_bias_gradient(neuron_count);
    m_weights_gradient(neuron_count, prev_layer_neuron_count);
    m_activations(neuron_count);
    m_activations_gradient(neuron_count, next_layer_neuron_count); //todo: revise row/columns assignment
}