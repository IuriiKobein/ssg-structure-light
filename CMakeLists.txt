cmake_minimum_required(VERSION 3.17)

project(3d_reconstruction CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${protobuf_VERSION}")

# gRPC
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")

find_package(cxxopts)

add_subdirectory(sl_alg)

add_executable(3dr "")
target_sources(3dr PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rpc_srv/tpu_service.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rpc_srv/proto_gen/sla.grpc.pb.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/rpc_srv/proto_gen/sla.pb.cc)
target_link_libraries(3dr sl_alg stdc++fs protobuf::libprotobuf
    gRPC::grpc++_unsecure)
