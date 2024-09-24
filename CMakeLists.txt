cmake_minimum_required(VERSION 3.29)
project(GestureRecognition)

set(CMAKE_CXX_STANDARD 23)

find_package(Eigen3 3.4 REQUIRED HINTS /opt/homebrew/opt/eigen)

add_executable(GestureRecognition main.cpp
        Layer.cpp
        Layer.h
        Training/ForwardPropagation.cpp
        Training/ForwardPropagation.h)

target_link_libraries(GestureRecognition Eigen3::Eigen)