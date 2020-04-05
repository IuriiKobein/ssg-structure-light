list(APPEND sl_alg_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/src/alg_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sl_pcg.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sl_tpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sl_opencv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sl_alg_factory.cpp
)

find_package(OpenCV)
list (APPEND sl_alg_libs ${OpenCV_LIBS})

find_package(CUDA)
if (CUDA_FOUND)
    add_subdirectory(cuda_kernels)
    list(APPEND sl_alg_srcs
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cu_sl_pcg.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cu_sl_tpu.cpp)
    list(APPEND sl_alg_libs cuda_kern)
endif(CUDA_FOUND)

add_library(sl_alg SHARED ${sl_alg_srcs})
target_include_directories(sl_alg
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(sl_alg PRIVATE -Werror)
target_link_libraries(sl_alg PUBLIC ${sl_alg_libs} stdc++fs )
