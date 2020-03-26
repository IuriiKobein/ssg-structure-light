#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <vector>

class sl_alg {
   public:
    sl_alg() {};
    virtual ~sl_alg() = default; 
    sl_alg& operator=(const sl_alg&) = delete;

    virtual int ref_phase_compute(const std::vector<cv::Mat> &refs) = 0;
    virtual cv::Mat depth_compute(const std::vector<cv::Mat> &objs) = 0;
};

