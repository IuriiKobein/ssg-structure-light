#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/types.hpp>
#include <string>
#include <vector>

class sl_alg {
   public:
    sl_alg() {};
    virtual ~sl_alg() = default; 
    sl_alg& operator=(const sl_alg&) = delete;

    virtual int ref_phase_compute(const std::vector<cv::cuda::GpuMat>& hf_ref) = 0;
    virtual int obj_phase_compute(const std::vector<cv::cuda::GpuMat>& hf_obj) = 0;
    virtual cv::Mat compute_3d_reconstruction() = 0;
};

