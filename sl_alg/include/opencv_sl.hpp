#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <string>
#include <vector>

#include "sl_alg.hpp"

class opencv_sl: public sl_alg {
   public:
    opencv_sl(const params_t& params);

    ~opencv_sl();

    virtual const std::vector<cv::Mat>& patterns_get();

    virtual int ref_phase_compute(
        const std::vector<cv::Mat>& refs);
    virtual cv::Mat depth_compute(const std::vector<cv::Mat>& objs);
    
    virtual int ref_phase_compute(const std::vector<cv::Mat> &lf_refs,
            const std::vector<cv::Mat>& hf_refs);
    virtual cv::Mat depth_compute(const std::vector<cv::Mat> &lf_objs,
            const std::vector<cv::Mat> &hf_objs);

   private:
    class alg_impl;
    std::unique_ptr<alg_impl> _pimpl;
};

