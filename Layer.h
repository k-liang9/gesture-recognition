#ifndef GESTURERECOGNITION_LAYER_H
#define GESTURERECOGNITION_LAYER_H
#include <Eigen/Dense>
#include <utility>

using Eigen::VectorXd;
using Eigen::MatrixXd;

enum activation {
    log_sigmoid,
    tan_sigmoid
};

class Layer {
private:
    VectorXd m_bias;
    VectorXd m_bias_gradient{};
    MatrixXd m_weights{};
    MatrixXd m_weights_gradient{};
    VectorXd m_activations{};
    MatrixXd m_activations_gradient{};
    int m_next_layer_neuron_count{};
    int m_neuron_count{};
    int m_prev_layer_neuron_count{};
    int m_layer_index{};

public:
    Layer(int layer_index, VectorXd bias, MatrixXd weights, int neuron_count, int prev_layer_neuron_count, int next_layer_neuron_count);

    [[nodiscard]] const VectorXd& bias() const { return m_bias; }
    void setBias(const VectorXd& bias) { m_bias = bias; }
    [[nodiscard]] const VectorXd& biasGradient() const { return m_bias_gradient; }
    void setBiasGradient(const VectorXd& bias_gradient) { m_bias_gradient = bias_gradient; }
    [[nodiscard]] const MatrixXd& weights() const { return m_weights; }
    void setWeights(const MatrixXd& weights) { m_weights = weights; }
    [[nodiscard]] const MatrixXd& weightsGradient() const { return m_weights_gradient; }
    void setWeightsGradient(const MatrixXd& weights_gradient) { m_weights_gradient = weights_gradient; }
    [[nodiscard]] const VectorXd& activations() const { return m_activations; }
    void setActivations(const VectorXd& activations) { m_activations = activations; }
    [[nodiscard]] const MatrixXd& activationsGradient() const { return m_activations_gradient; }
    void setActivationsGradient(const MatrixXd& activations_gradient) { m_activations_gradient = activations_gradient; }
    [[nodiscard]] int nextLayerNeuronCount() const { return m_next_layer_neuron_count; }
    void setNextLayerNeuronCount(int next_layer_neuron_count) { m_next_layer_neuron_count = next_layer_neuron_count; }
    [[nodiscard]] int neuronCount() const { return m_neuron_count; }
    void setNeuronCount(int neuron_count) { m_neuron_count = neuron_count; }
    [[nodiscard]] int prevLayerNeuronCount() const { return m_prev_layer_neuron_count; }
    void setPrevLayerNeuronCount(int prev_layer_neuron_count) { m_prev_layer_neuron_count = prev_layer_neuron_count; }
    [[nodiscard]] int layerIndex() const { return m_layer_index; }
    void setLayerIndex(int layer_index) { m_layer_index = layer_index; }
};


#endif //GESTURERECOGNITION_LAYER_H
