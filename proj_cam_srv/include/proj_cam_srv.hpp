#pragma once
#include <vector>
#include <opencv2/videoio.hpp>

using img_vec_t = std::vector<cv::Mat>;

// helper class for project and capture image for sl algorithm
struct proj_cam_srv {
    proj_cam_srv();

    void size_set(const cv::Size& size);
    void capture_timeout_set(int timeout_ms);

    void images_capture(const img_vec_t::const_iterator patterns_begin_it,
                            const img_vec_t::const_iterator patterns_end_it,
                            img_vec_t::iterator capture_it); 
   private:
    std::string _win_name;
    std::int32_t _capture_timeout_ms;
    cv::VideoCapture _cap;
};

