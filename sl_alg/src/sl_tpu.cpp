#include "sl_tpu.hpp"

#include "alg_utils.hpp"
#include "sl_alg_factory.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/cudafilters.hpp>

#include <algorithm>
#include <iostream>
#include <vector>

namespace {
sl_alg_auto_reg s_sl_tpu_reg("cpu_tpu", [](cv::Size size) {
    return std::make_unique<sl_tpu>(size);
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
    alg_impl(cv::Size size)
        : _params{20, 1},
          _lf_obj_phase(size, CV_32FC1),
          _hf_obj_phase(size, CV_32FC1),
          _lf_ref_phase(size, CV_32FC1),
          _hf_ref_phase(size, CV_32FC1)

    {}

    int tpu_config_set(const tpu_params_t& params) {
        _params = params;
        return 0;
    }

    int ref_phase_compute(const std::vector<cv::Mat>& lf_refs,
                          const std::vector<cv::Mat>& hf_refs) {
        cpu_phase_compute(lf_refs, _lf_ref_phase);
        cpu_phase_compute(hf_refs, _hf_ref_phase);

        return 0;
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
    tpu_params_t _params;
    std::vector<cv::Mat> _tmp;
    cv::Mat _lf_obj_phase;
    cv::Mat _hf_obj_phase;
    cv::Mat _lf_ref_phase;
    cv::Mat _hf_ref_phase;
};

sl_tpu::sl_tpu(cv::Size size) : _pimpl(std::make_unique<alg_impl>(size)) {}

sl_tpu::~sl_tpu() = default;

int sl_tpu::ref_phase_compute(const std::vector<cv::Mat>& refs) {
    return -ENOTSUP;
}

cv::Mat sl_tpu::depth_compute(const std::vector<cv::Mat>& objs) {
    return cv::Mat();
}

int sl_tpu::ref_phase_compute(const std::vector<cv::Mat>& lf_refs,
                              const std::vector<cv::Mat>& hf_refs) {
    return _pimpl->ref_phase_compute(lf_refs, hf_refs);
}

cv::Mat sl_tpu::depth_compute(const std::vector<cv::Mat>& lf_objs,
                              const std::vector<cv::Mat>& hf_objs) {
    return _pimpl->depth_compute(lf_objs, hf_objs);
}
