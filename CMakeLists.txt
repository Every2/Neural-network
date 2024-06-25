cmake_minimum_required(VERSION 3.28)
project(neural_network)

set(CMAKE_CXX_STANDARD 20)

set(SOURCES
        src/main.cc
        include/network.hh
        src/network.cc

)
add_executable(neural_network ${SOURCES})