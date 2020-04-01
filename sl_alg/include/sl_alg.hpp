#pragma once

#include <cmath>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <vector>

class sl_alg {
   public:
    struct params_t {
        cv::Size size;
        std::float_t freq_ratio;
        std::float_t real_scale;
        std::int32_t num_of_periods;
        std::int32_t use_markers;
        std::int32_t is_horizontal;
        std::int32_t num_of_patterns;
        std::int32_t num_of_pix_mark;
        std::int32_t opencv_method_id;
    };
    sl_alg() {};
    virtual ~sl_alg() = default; 
    sl_alg& operator=(const sl_alg&) = delete;

    virtual const std::vector<cv::Mat>& patterns_get() = 0;

    virtual int ref_phase_compute(const std::vector<cv::Mat> &refs) = 0;
    virtual cv::Mat depth_compute(const std::vector<cv::Mat> &objs) = 0;

    virtual int ref_phase_compute(const std::vector<cv::Mat> &lf_refs,
            const std::vector<cv::Mat>& hf_refs) = 0;
    virtual cv::Mat depth_compute(const std::vector<cv::Mat> &lf_objs,
            const std::vector<cv::Mat> &hf_objs) = 0;
};

