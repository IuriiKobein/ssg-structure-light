cmake_minimum_required(VERSION 3.0)

project(3d_reconstruction CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.2/bin/nvcc)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -g -O2")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_subdirectory(pu_alg)

set(3dr_srcs main.cpp)
add_executable(3dr ${3dr_srcs})
target_link_libraries(3dr pu_alg)
target_link_libraries(3dr -lstdc++fs)
