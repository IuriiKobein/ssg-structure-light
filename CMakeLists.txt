cmake_minimum_required(VERSION 3.0)

project(3d_reconstruction CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -g -O2")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(cxxopts)

add_subdirectory(pu_alg)

add_executable(3dr "")
target_sources(3dr PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
target_link_libraries(3dr pu_alg stdc++fs)
