#include <iostream>
#include "Layers/Dense.h"
#include "Training/ForwardPropagation.h"
#include <Eigen/Dense>

/*
 * 	1.	Conv2D(32, (3,3), activation=‘relu’)
	2.	MaxPooling2D((2,2))
	3.	Conv2D(64, (3,3), activation=‘relu’)
	4.	MaxPooling2D((2,2))
	5.	Conv2D(128, (3,3), activation=‘relu’)
	6.	Flatten
	7.	Dense(128, activation=‘relu’)
	8.	Dropout(0.25)
	9.	Dense(num_classes, activation=‘softmax’)
 */

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}