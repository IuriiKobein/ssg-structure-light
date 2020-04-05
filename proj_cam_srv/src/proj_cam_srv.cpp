#include "proj_cam_srv.hpp"
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

// helper class for project and capture image for sl algorithm
proj_cam_srv::proj_cam_srv()
    : _win_name("structured_light_capture"),
      _capture_timeout_ms(300),
      _cap(0, cv::CAP_V4L2) {
    if (!_cap.isOpened()) {
        std::cout << "Camera could not be opened\n";
    }
    cv::namedWindow(_win_name, cv::WINDOW_NORMAL);
    cv::setWindowProperty(_win_name, cv::WND_PROP_FULLSCREEN,
                          cv::WINDOW_FULLSCREEN);
}

void proj_cam_srv::size_set(const cv::Size& size) {
    std::cout << _cap.set(cv::CAP_PROP_FRAME_WIDTH, size.width);
    std::cout << _cap.set(cv::CAP_PROP_FRAME_HEIGHT, size.height);
}

void proj_cam_srv::capture_timeout_set(int timeout_ms) {
    _capture_timeout_ms = timeout_ms;
}

void proj_cam_srv::images_capture(
    const img_vec_t::const_iterator patterns_begin_it,
    const img_vec_t::const_iterator patterns_end_it,
    img_vec_t::iterator capture_it) {
    std::for_each(patterns_begin_it, patterns_end_it,
                  [this, &capture_it](const auto& pattern) {
                      cv::Mat rgb;
                      cv::imshow(_win_name, pattern);
                      cv::waitKey(_capture_timeout_ms);
                      _cap >> rgb;
                      cv::cvtColor(rgb, *capture_it, cv::COLOR_BGR2GRAY);
                      ++capture_it;
                  });
}

