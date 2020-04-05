list(APPEND proj_cam_srv_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/src/proj_cam_srv.cpp
)

find_package(OpenCV)

add_library(proj_cam_srv SHARED ${proj_cam_srv_srcs})

target_include_directories(proj_cam_srv
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)


target_compile_options(proj_cam_srv PRIVATE -Werror)
target_link_libraries(proj_cam_srv PUBLIC ${OpenCV_LIBS})
