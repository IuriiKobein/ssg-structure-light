add_subdirectory(cuda_kernels)

set(pu_alg_srcs
    src/alg_utils.cpp
    src/sl_pcg_alg.cpp
    src/sl_tpu_alg.cpp
    src/cuda_phase_unwrap.cpp)

add_library(pu_alg SHARED ${pu_alg_srcs})
target_include_directories(pu_alg PUBLIC include PRIVATE src)
target_link_libraries(pu_alg PUBLIC cuda_kern)
