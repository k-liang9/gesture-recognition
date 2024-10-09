#include <iostream>
#include "Layers/Dense.h"
#include "Training/ForwardPropagation.h"
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
	10.	Dense(128, activation=‘relu’)
	11.	Dropout(0.2)
	12.	Dense(num_classes, do_activation=‘softmax’)
 */

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}