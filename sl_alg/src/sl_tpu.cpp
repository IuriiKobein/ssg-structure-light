#include "sl_tpu.hpp"

#include "alg_utils.hpp"
#include "sl_alg_factory.hpp"

#include <iterator>
#include <opencv2/core.hpp>

#include <algorithm>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <vector>

namespace {
sl_alg_auto_reg s_sl_tpu_reg("cpu_tpu", [](const sl_alg::params_t& params) {
    return std::make_unique<sl_tpu>(params);
});

cv::Mat cpu_temporal_phase_unwrap(cv::Mat& phase1, cv::Mat& phase2,
                                  float scale) {
    cv::Mat out;

    phase1.convertTo(phase1, phase1.type(), scale);
    cv::subtract(phase1, phase2, phase1);
    phase1.convertTo(phase1, phase1.type(), 0.5f / CV_PI);
    cv_round(phase1);
    phase1.convertTo(phase1, phase1.type(), 2 * CV_PI);
    cv::add(phase1, phase2, out);

    return out;
}
}  // namespace

class sl_tpu::alg_impl {
   public:
    alg_impl(const params_t& params)
        : _params(params),
          _lf_obj_phase(_params.size, CV_32FC1),
          _hf_obj_phase(_params.size, CV_32FC1),
          _lf_ref_phase(_params.size, CV_32FC1),
          _hf_ref_phase(_params.size, CV_32FC1)

    {
        sinusoidal_pattern_params sinus_params;

        // 1 period low frequency pattern for TPU specific approach 
        sinus_params.is_horizontal = params.is_horizontal;
        sinus_params.num_of_patterns = 1;
        sinus_params.num_of_periods = params.num_of_periods;
        sinus_params.size = params.size;
        _patterns = sinusoidal_pattern_generate(sinus_params);
        
        // regular high frequency patterns
        sinus_params.is_horizontal = params.is_horizontal;
        sinus_params.num_of_patterns = params.num_of_patterns;
        sinus_params.num_of_periods = params.num_of_periods;
        sinus_params.size = params.size;
        auto hf_patterns = sinusoidal_pattern_generate(sinus_params);

        _patterns.insert(std::begin(_patterns), std::begin(hf_patterns), std::end(hf_patterns));
    }

    const std::vector<cv::Mat>& patterns_get() { return _patterns; }

    cv::Mat ref_phase_compute(const std::vector<cv::Mat>& lf_refs,
                          const std::vector<cv::Mat>& hf_refs) {
        cpu_phase_compute(lf_refs, _lf_ref_phase);
        cpu_phase_compute(hf_refs, _hf_ref_phase);

        return _hf_ref_phase;
    }

    cv::Mat depth_compute(const std::vector<cv::Mat>& lf_objs,
                          const std::vector<cv::Mat>& hf_objs) {
        cpu_phase_compute(lf_objs, _lf_obj_phase);
        cpu_phase_compute(hf_objs, _hf_obj_phase);

        cv::subtract(_lf_obj_phase, _lf_ref_phase, _lf_obj_phase);
        cv::subtract(_hf_obj_phase, _hf_ref_phase, _hf_obj_phase);

        return cpu_temporal_phase_unwrap(_lf_obj_phase, _hf_obj_phase,
                                         _params.freq_ratio);
    }

   private:
    params_t _params;
    std::vector<cv::Mat> _patterns;
    std::vector<cv::Mat> _tmp;
    cv::Mat _lf_obj_phase;
    cv::Mat _hf_obj_phase;
    cv::Mat _lf_ref_phase;
    cv::Mat _hf_ref_phase;
};

sl_tpu::sl_tpu(const params_t& params)
    : _pimpl(std::make_unique<alg_impl>(params)) {}

sl_tpu::~sl_tpu() = default;

const std::vector<cv::Mat>& sl_tpu::patterns_get() {
    return _pimpl->patterns_get();
}

cv::Mat sl_tpu::ref_phase_compute(const std::vector<cv::Mat>& refs) {
    return cv::Mat();
}

cv::Mat sl_tpu::depth_compute(const std::vector<cv::Mat>& objs) {
    return cv::Mat();
}

cv::Mat sl_tpu::ref_phase_compute(const std::vector<cv::Mat>& lf_refs,
                              const std::vector<cv::Mat>& hf_refs) {
    return _pimpl->ref_phase_compute(lf_refs, hf_refs);
}

cv::Mat sl_tpu::depth_compute(const std::vector<cv::Mat>& lf_objs,
                              const std::vector<cv::Mat>& hf_objs) {
    return _pimpl->depth_compute(lf_objs, hf_objs);
}
