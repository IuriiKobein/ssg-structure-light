list(APPEND sl_rpc_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lfs_transport.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rpc_server.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sla_ctrl_service.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/proto_gen/sla.grpc.pb.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/proto_gen/sla.pb.cc
)

add_library(sl_rpc SHARED ${sl_rpc_srcs})

target_include_directories(sl_rpc
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/proto_gen
)

target_compile_options(sl_rpc PRIVATE -Werror)
target_link_libraries(sl_rpc PUBLIC sl_alg protobuf::libprotobuf
    gRPC::grpc++_unsecure )
