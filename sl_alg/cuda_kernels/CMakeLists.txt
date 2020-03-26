find_package(CUDA)
find_package(OpenCV)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH} -g")

add_library(cuda_kern SHARED src/cuda_kernels.cu src/cu_alg_utils.cpp)
target_include_directories(cuda_kern PUBLIC include ${CUDA_INCLUDE_DIRS} PRIVATE src)
set_target_properties(cuda_kern PROPERTIES CUDA_STANDARD 14
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cuda_kern PUBLIC ${CUDA_LIBS} ${OpenCV_LIBS})
