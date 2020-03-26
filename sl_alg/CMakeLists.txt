list(APPEND sl_alg_srcs
    ${CMAKE_CURRENT_LIST_DIR}/src/alg_utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sl_pcg.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sl_alg_factory.cpp
)

find_package(OpenCV)
list (APPEND sl_alg_libs ${OpenCV_LIBS})

find_package(CUDA)
if (CUDA_FOUND)
    add_subdirectory(cuda_kernels)
    list(APPEND sl_alg_srcs
        ${CMAKE_CURRENT_LIST_DIR}/src/cu_sl_pcg.cpp)
    list(APPEND sl_alg_libs cuda_kern)
endif(CUDA_FOUND)

add_library(sl_alg SHARED ${sl_alg_srcs})
target_include_directories(sl_alg PUBLIC include PRIVATE src)
target_link_libraries(sl_alg PUBLIC ${sl_alg_libs})



