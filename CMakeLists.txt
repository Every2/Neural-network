cmake_minimum_required(VERSION 3.28)
project(neural_network)

set(CMAKE_CXX_STANDARD 20)

set(SOURCES
        src/main.cpp
        include/network.hpp
        src/network.cpp

)
add_executable(neural_network ${SOURCES})