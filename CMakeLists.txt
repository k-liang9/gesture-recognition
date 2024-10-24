cmake_minimum_required(VERSION 3.29)
project(GestureRecognition)

set(CMAKE_CXX_STANDARD 23)

find_package(Eigen3 3.4 REQUIRED HINTS /opt/homebrew/opt/eigen)

add_executable(GestureRecognition main.cpp
        Layers/DenseL/DenseNormalization.cpp
        Layers/DenseL/DenseL.h
        Global.h
        Layers/Layer.cpp
        Layers/Layer.h
        Layers/ConvL/ConvL.cpp
        Layers/ConvL/ConvL.h
        Layers/DenseL/DenseForwardPropagation.cpp
        Layers/DenseL/DenseBackPropagation.cpp
        Layers/DenseL/DenseMisc.cpp)

target_link_libraries(GestureRecognition Eigen3::Eigen)