#include "cu_sl_tpu.hpp"
#include "cu_alg_utils.hpp"
#include "cuda_kernels.h"

#include "alg_utils.hpp"
#include "sl_alg_factory.hpp"

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/cudafilters.hpp>

#include <algorithm>
#include <iostream>
#include <vector>

namespace {
sl_alg_auto_reg s_cu_sl_tpu_reg("cuda_tpu", [](const sl_alg::params_t& params) {
    return std::make_unique<cu_sl_tpu>(params);
});

cv::Mat cuda_temporal_phase_unwrap(cv::cuda::GpuMat& phase1,
                                   cv::cuda::GpuMat& phase2, float scale) {
    cv::Mat out;

    //cuda_wrap(phase1);
    cuda_temporal_unwrap(phase1, phase2, phase1, scale);

    phase1.download(out);

    return out;
}
}  // namespace

class cu_sl_tpu::alg_impl {
   public:
    alg_impl(const params_t& params)
        : _params(params),
          _tmp(imgs_alloc(4, _params.size, CV_32FC1)),
          _cu_tmp(cuda_imgs_alloc(4, _params.size, CV_32FC1)),
          _lf_obj_phase(cuda_img_alloc(_params.size, CV_32FC1)),
          _hf_obj_phase(cuda_img_alloc(_params.size, CV_32FC1)),
          _lf_ref_phase(cuda_img_alloc(_params.size, CV_32FC1)),
          _hf_ref_phase(cuda_img_alloc(_params.size, CV_32FC1)),
          _filt(cv::cuda::createGaussianFilter(CV_32F, CV_32F, cv::Size(3, 3),
                                               0)) {
        sinusoidal_pattern_params sinus_params;

        // 1 period low frequency pattern for TPU specific approach
        sinus_params.is_horizontal = params.is_horizontal;
        sinus_params.num_of_patterns = params.num_of_patterns;
        sinus_params.num_of_periods = 1;
        sinus_params.size = params.size;
        _patterns = sinusoidal_pattern_generate(sinus_params);

        // regular high frequency patterns
        sinus_params.is_horizontal = params.is_horizontal;
        sinus_params.num_of_patterns = params.num_of_patterns;
        sinus_params.num_of_periods = params.num_of_periods;
        sinus_params.size = params.size;
        auto hf_patterns = sinusoidal_pattern_generate(sinus_params);

        _patterns.insert(std::end(_patterns), std::begin(hf_patterns),
                         std::end(hf_patterns));
    }

    const std::vector<cv::Mat>& patterns_get() { return _patterns; }

    cv::Mat ref_phase_compute(const std::vector<cv::Mat>& lf_refs,
                          const std::vector<cv::Mat>& hf_refs) {
        cv::Mat cpu;

        cuda_phase_compute(lf_refs, _tmp, _cu_tmp, *_filt, _lf_ref_phase);
        cuda_phase_compute(hf_refs, _tmp, _cu_tmp, *_filt, _hf_ref_phase);

        _hf_ref_phase.download(cpu);

        return cpu;
    }

    cv::Mat depth_compute(const std::vector<cv::Mat>& lf_objs,
                          const std::vector<cv::Mat>& hf_objs) {
        auto ts = std::chrono::high_resolution_clock::now();
        cuda_phase_compute(lf_objs, _tmp, _cu_tmp, *_filt, _lf_obj_phase);
        cuda_phase_compute(hf_objs, _tmp, _cu_tmp, *_filt, _hf_obj_phase);
        auto te = std::chrono::high_resolution_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(te -
                                                                           ts)
                         .count()
                  << std::endl;

        cv::cuda::subtract(_lf_obj_phase, _lf_ref_phase, _lf_obj_phase);
        cv::cuda::subtract(_hf_obj_phase, _hf_ref_phase, _hf_obj_phase);

        return cuda_temporal_phase_unwrap(_lf_obj_phase, _hf_obj_phase,
                                          _params.freq_ratio);
    }

   private:
    params_t _params;
    std::vector<cv::Mat> _patterns;
    std::vector<cv::Mat> _tmp;
    std::vector<cv::cuda::GpuMat> _cu_tmp;
    cv::cuda::GpuMat _lf_obj_phase;
    cv::cuda::GpuMat _hf_obj_phase;
    cv::cuda::GpuMat _lf_ref_phase;
    cv::cuda::GpuMat _hf_ref_phase;
    cv::Ptr<cv::cuda::Filter> _filt;
};

cu_sl_tpu::cu_sl_tpu(const params_t& params)
    : _pimpl(std::make_unique<alg_impl>(params)) {}

cu_sl_tpu::~cu_sl_tpu() = default;

const std::vector<cv::Mat>& cu_sl_tpu::patterns_get() {
    return _pimpl->patterns_get();
}

cv::Mat cu_sl_tpu::ref_phase_compute(const std::vector<cv::Mat>& refs) {
    return cv::Mat();
}

cv::Mat cu_sl_tpu::depth_compute(const std::vector<cv::Mat>& objs) {
    return cv::Mat();
}

cv::Mat cu_sl_tpu::ref_phase_compute(const std::vector<cv::Mat>& lf_refs,
                                 const std::vector<cv::Mat>& hf_refs) {
    return _pimpl->ref_phase_compute(lf_refs, hf_refs);
}

cv::Mat cu_sl_tpu::depth_compute(const std::vector<cv::Mat>& lf_objs,
                                 const std::vector<cv::Mat>& hf_objs) {
    return _pimpl->depth_compute(lf_objs, hf_objs);
}
