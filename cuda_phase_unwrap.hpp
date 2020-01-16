#pragma once

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/types.hpp>

#include <memory>

class cuda_phase_unwrap_alg {
   public:
    cuda_phase_unwrap_alg(cv::Size size);
    ~cuda_phase_unwrap_alg();

    cv::Mat compute(cv::cuda::GpuMat& in);

   private:
    class cu_pu_impl;
    std::unique_ptr<cu_pu_impl> _pimpl;
};
