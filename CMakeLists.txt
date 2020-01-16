cmake_minimum_required(VERSION 3.0)
project(OpenCVProject)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -g -O2")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
enable_language("CUDA")

find_package(OpenCV)

find_package(CUDA)
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH} -g")

include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} src)

set(3dr_srcs main.cpp
    alg_utils.cpp
    cl_alg.cpp
    cuda_phase_unwrap.cpp)

add_library(cuda_alg SHARED cuda_kernels.cu)
set_property(TARGET cuda_alg PROPERTY CUDA_STANDARD 14)
set_target_properties(cuda_alg PROPERTIES POSITION_INDEPENDENT_CODE ON)
#set_target_properties(CudaLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(3dr ${3dr_srcs})

target_link_libraries(3dr ${OpenCV_LIBS})
target_link_libraries(3dr ${CUDA_LIBS})
target_link_libraries(3dr cuda_alg)
target_link_libraries(3dr -lstdc++fs)
