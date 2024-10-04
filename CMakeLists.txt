cmake_minimum_required(VERSION 3.29)
project(GestureRecognition)

set(CMAKE_CXX_STANDARD 23)

find_package(Eigen3 3.4 REQUIRED HINTS /opt/homebrew/opt/eigen)

add_executable(GestureRecognition main.cpp
        Layers/Dense.cpp
        Layers/Dense.h
        Training/ForwardPropagation.cpp
        Training/ForwardPropagation.h
        Global.h
        Layers/Layer.cpp
        Layers/Layer.h
        Layers/Conv.cpp
        Layers/Conv.h)

target_link_libraries(GestureRecognition Eigen3::Eigen)