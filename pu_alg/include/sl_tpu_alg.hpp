#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/types.hpp>
#include <string>
#include <vector>

#include "sl_alg.hpp"

class sl_tpu_alg:public sl_alg {
   public:
    sl_tpu_alg(cv::Size size);
    ~sl_tpu_alg();

    virtual int ref_phase_compute(const std::vector<cv::cuda::GpuMat>& imgs) override;
    virtual int obj_phase_compute(const std::vector<cv::cuda::GpuMat>& imgs) override;
    virtual cv::Mat compute_3d_reconstruction() override;

   private:
    class alg_impl;
    std::unique_ptr<alg_impl> _pimpl;
};

