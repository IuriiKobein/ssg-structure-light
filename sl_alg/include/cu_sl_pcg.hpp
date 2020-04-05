#pragma once

#include "sl_alg.hpp"
#include <opencv2/core/mat.hpp>

class cu_sl_pcg : public sl_alg {
   public:
    cu_sl_pcg(const params_t& params);
    ~cu_sl_pcg();

    virtual const std::vector<cv::Mat>& patterns_get();

    virtual cv::Mat ref_phase_compute(const std::vector<cv::Mat> &refs);
    virtual cv::Mat depth_compute(const std::vector<cv::Mat> &objs);

    virtual cv::Mat ref_phase_compute(const std::vector<cv::Mat> &lf_refs,
            const std::vector<cv::Mat>& hf_refs);
    virtual cv::Mat depth_compute(const std::vector<cv::Mat> &lf_objs,
            const std::vector<cv::Mat> &hf_objs);

   private:
    class cu_sl_pcg_impl;
    std::unique_ptr<cu_sl_pcg_impl> _pimpl;
};

