cmake_minimum_required(VERSION 3.13)

set(proj_langs CXX)

project(3d_reconstruction CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${protobuf_VERSION}")

# gRPC
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")

add_subdirectory(proj_cam_srv)
add_subdirectory(rpc_srv)
add_subdirectory(sl_alg)

add_executable(3dr "")
target_compile_options(3dr PRIVATE -Werror)
target_sources(3dr PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/experimental_sl_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/opencv_sl_main.cpp
)

target_link_libraries(3dr sl_rpc  )
