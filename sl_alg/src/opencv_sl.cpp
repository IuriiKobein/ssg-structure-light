#include "opencv_sl.hpp"

#include "alg_utils.hpp"
#include "sl_alg_factory.hpp"

#include <opencv2/core.hpp>
#include <opencv2/phase_unwrapping.hpp>
#include <opencv2/structured_light.hpp>

#include <algorithm>
#include <iostream>
#include <opencv2/structured_light/sinusoidalpattern.hpp>
#include <vector>

namespace {
sl_alg_auto_reg s_opencv_sl_reg("opencv_sl",
                                [](const sl_alg::params_t& params) {
                                    return std::make_unique<opencv_sl>(params);
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

class opencv_sl::alg_impl {
   public:
    alg_impl(const params_t& params) : _params(params) {
        _sl_params.width = params.size.width;
        _sl_params.height = params.size.height;
        _sl_params.nbrOfPeriods = params.num_of_periods;
        _sl_params.setMarkers = params.use_markers;
        _sl_params.horizontal = params.is_horizontal;
        _sl_params.methodId = params.opencv_method_id;
        _sl_params.shiftValue = static_cast<float>(2 * CV_PI / 3);
        _sl_params.nbrOfPixelsBetweenMarkers = 70;
        _sl_alg = cv::structured_light::SinusoidalPattern::create(
            cv::makePtr<cv::structured_light::SinusoidalPattern::Params>(
                _sl_params));

        _pu_params.height = params.size.height;
        _pu_params.width = params.size.width;
        _pu_alg =
            cv::phase_unwrapping::HistogramPhaseUnwrapping::create(_pu_params);

        _sl_alg->generate(_patterns);
    }

    const std::vector<cv::Mat>& patterns_get() { return _patterns; }

    cv::Mat depth_compute(const std::vector<cv::Mat>& lf_objs) {
        cv::Mat reliabilities;

        _sl_alg->unwrapPhaseMap(_wpm, _upm, _img_size, _shadow_mask);
        _pu_alg->unwrapPhaseMap(_upm, _wpm, _shadow_mask);

        _pu_alg->getInverseReliabilityMap(reliabilities);

        return reliabilities;
    }

   private:
    params_t _params;
    cv::Size _img_size;
    std::vector<cv::Mat> _patterns;
    cv::structured_light::SinusoidalPattern::Params _sl_params;
    cv::phase_unwrapping::HistogramPhaseUnwrapping::Params _pu_params;
    cv::Ptr<cv::structured_light::SinusoidalPattern> _sl_alg;
    cv::Ptr<cv::phase_unwrapping::HistogramPhaseUnwrapping> _pu_alg;
    cv::Mat _shadow_mask;
    cv::Mat _upm;
    cv::Mat _wpm;
};

opencv_sl::opencv_sl(const params_t& params)
    : _pimpl(std::make_unique<alg_impl>(params)) {}

opencv_sl::~opencv_sl() = default;

const std::vector<cv::Mat>& opencv_sl::patterns_get() {
    return _pimpl->patterns_get();
}

int opencv_sl::ref_phase_compute(const std::vector<cv::Mat>& refs) {
    return -ENOTSUP;
}

cv::Mat opencv_sl::depth_compute(const std::vector<cv::Mat>& objs) {
    return _pimpl->depth_compute(objs);
}

int opencv_sl::ref_phase_compute(const std::vector<cv::Mat>& lf_refs,
                                 const std::vector<cv::Mat>& hf_refs) {
    return -ENOTSUP;
}

cv::Mat opencv_sl::depth_compute(const std::vector<cv::Mat>& lf_objs,
                                 const std::vector<cv::Mat>& hf_objs) {
    return cv::Mat();
}
