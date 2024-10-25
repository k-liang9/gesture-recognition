#include <iostream>
#include "Layers/DenseL/DenseL.h"
#include <Eigen/Dense>

/*
 * 	1.	Conv2D(8, (3, 3), activation=‘relu’)
	2.	MaxPooling2D((2, 2))
	3.	Conv2D(16, (3, 3), activation=‘relu’)
	4.	MaxPooling2D((2, 2))
	5.	Conv2D(32, (3, 3), activation=‘relu’)
	6.	MaxPooling2D((2, 2))
	7.	Conv2D(64, (3, 3), activation=‘relu’)
	8.	MaxPooling2D((2, 2))
	9.	Flatten()
	10.	DenseL(128, activation=‘relu’)
	11.	Dropout(0.2)
	12.	DenseL(num_classes, calc_activation=‘softmax’)

 Mini-batch training: 32
 */

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}