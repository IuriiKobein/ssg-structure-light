#pragma once
#include <opencv2/core/types.hpp>
#include <vector>
#include <opencv2/videoio.hpp>

using img_vec_t = std::vector<cv::Mat>;

// helper class for project and capture image for sl algorithm
struct proj_cam_srv {
    proj_cam_srv();

    void size_set(const cv::Size& size);
    cv::Size size_get() const;

    void capture_timeout_set(int timeout_ms);
    int capture_timeout_get() const;

    void images_capture(const img_vec_t::const_iterator patterns_begin_it,
                            const img_vec_t::const_iterator patterns_end_it,
                            img_vec_t::iterator capture_it); 
   private:
    std::string _win_name;
    std::int32_t _capture_timeout_ms;
    cv::VideoCapture _cap;
};

