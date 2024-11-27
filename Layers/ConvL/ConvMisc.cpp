#include "ConvL.h"
#include "Eigen/Dense"
#include <unsupported/Eigen/CXX11/Tensor>
#include <cassert>
#include <iostream>

using namespace Eigen;

ConvL::ConvL(Tensor<double, 4> *f, VectorXd* b, activation a, bool last) :
        filter(*f),
        biases(*b),
        is_last{last}
{
    set_activation_func(a);
    gradient_sum_filter.resize(filter.dimensions());
    gradient_sum_biases.resize(filter.size());
}

void ConvL::convolve(const Tensor<double, 2>& input, const Tensor<double, 2>& kernel,
                        Tensor<double, 2>& output) {
    int input_rows = input.dimension(0);
    int input_cols = input.dimension(1);
    int kernel_rows = kernel.dimension(0);
    int kernel_cols = kernel.dimension(1);
    int output_rows = output.dimension(0);
    int output_cols = output.dimension(1);

    assert(input_rows >= kernel_rows);
    assert(input_cols >= kernel_cols);
    assert(output_rows == input_rows - kernel_rows + 1);
    assert(output_cols == input_cols - kernel_cols + 1);

    for (int row = 0; row <= input_rows - kernel_rows; ++row) {
        for (int col = 0; col <= input_cols - kernel_cols; ++col) {
            Tensor<double, 2> input_slice = input.slice
                    (Eigen::array<long, 2>{row, col}, kernel.dimensions());
            Tensor<double, 1> result = (kernel * input_slice).sum();
            output(row, col) += result(0);
        }
    }
}

void ConvL::convolve_full(const Tensor<double, 2>& input, const Tensor<double, 2>& kernel,
                          Tensor<double, 2>& output) {
    int input_rows = input.dimension(0);
    int input_cols = input.dimension(1);
    int kernel_rows = kernel.dimension(0);
    int kernel_cols = kernel.dimension(1);
    int output_rows = output.dimension(0);
    int output_cols = output.dimension(1);

    assert(output_rows == input_rows + kernel_rows - 1);
    assert(output_cols == input_cols + kernel_cols - 1);

    for (int output_row = 0; output_row < output_rows; ++output_row) {
        for (int output_col = 0; output_col < output_cols; ++output_col) {

            double sum = 0;
            for (int kernel_row = 0; kernel_row < kernel_rows; ++kernel_row) {
                for (int kernel_col = 0; kernel_col < kernel_cols; ++kernel_col) {


                    int input_row = output_row + kernel_row - (kernel_rows-1);
                    int input_col = output_col + kernel_col - (kernel_cols-1);

                    //padding
                    double input_val = 0;
                    if (input_row >= 0 && input_row < input_rows &&
                        input_col >= 0 && input_col < input_cols) {
                        input_val = input(input_row, input_col);
                    }

                    sum += input_val * kernel(kernel_row, kernel_col);
                }
            }

            output(output_row, output_col) += sum;
        }
    }
}