#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <string>
#include <vector>

#include "sl_alg.hpp"

class sl_tpu: public sl_alg {
   public:
    struct tpu_params_t {
        std::uint32_t freq_ratio;
        std::float_t real_scale;
    };
    sl_tpu(cv::Size size);
    ~sl_tpu();

    int tpu_config_set(const tpu_params_t &params);
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

