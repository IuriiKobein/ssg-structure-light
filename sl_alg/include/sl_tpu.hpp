#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "sl_alg.hpp"

class sl_tpu: public sl_alg {
   public:
    sl_tpu(const params_t& params);
    ~sl_tpu();

    virtual const std::vector<cv::Mat>& patterns_get();

    virtual cv::Mat ref_phase_compute(
        const std::vector<cv::Mat>& refs);
    virtual cv::Mat depth_compute(const std::vector<cv::Mat>& objs);
    
    virtual cv::Mat ref_phase_compute(const std::vector<cv::Mat> &lf_refs,
            const std::vector<cv::Mat>& hf_refs);
    virtual cv::Mat depth_compute(const std::vector<cv::Mat> &lf_objs,
            const std::vector<cv::Mat> &hf_objs);

   private:
    class alg_impl;
    std::unique_ptr<alg_impl> _pimpl;
};

