#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/types.hpp>
#include <string>
#include <vector>

class structure_light_alg {
   public:
    structure_light_alg(cv::Size size);
    ~structure_light_alg();
    structure_light_alg& operator=(const structure_light_alg&) = delete;
    structure_light_alg& operator=(structure_light_alg&&) = delete;

    int ref_phase_compute(const std::vector<cv::cuda::GpuMat>& imgs);
    cv::Mat compute_3d_reconstruction(const std::vector<cv::cuda::GpuMat>& imgs);

   private:
    class sl_alg_impl;
    std::unique_ptr<sl_alg_impl> _pimpl;
};

