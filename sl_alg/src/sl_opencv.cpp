#include "sl_opencv.hpp"

#include "alg_utils.hpp"
#include "sl_alg_factory.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/phase_unwrapping.hpp>
#include <opencv2/structured_light.hpp>

#include <algorithm>
#include <iostream>
#include <opencv2/structured_light/sinusoidalpattern.hpp>
#include <string>
#include <vector>

namespace {
sl_alg_auto_reg s_sl_opencv_reg("sl_opencv",
                                [](const sl_alg::params_t& params) {
                                    return std::make_unique<sl_opencv>(params);
                                });
}  // namespace

class sl_opencv::alg_impl {
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
        _img_size = params.size;
    }

    const std::vector<cv::Mat>& patterns_get() { return _patterns; }

    cv::Mat depth_compute(const std::vector<cv::Mat>& hf_objs) {
        _sl_alg->computePhaseMap(hf_objs, _wpm, _shadow_mask);
        _sl_alg->unwrapPhaseMap(_wpm, _upm, _img_size, _shadow_mask);

        return _upm;
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

sl_opencv::sl_opencv(const params_t& params)
    : _pimpl(std::make_unique<alg_impl>(params)) {}

sl_opencv::~sl_opencv() = default;

const std::vector<cv::Mat>& sl_opencv::patterns_get() {
    return _pimpl->patterns_get();
}

cv::Mat sl_opencv::ref_phase_compute(const std::vector<cv::Mat>& refs) {
    return cv::Mat();
}

cv::Mat sl_opencv::depth_compute(const std::vector<cv::Mat>& objs) {
    return _pimpl->depth_compute(objs);
}

cv::Mat sl_opencv::ref_phase_compute(const std::vector<cv::Mat>& lf_refs,
                                 const std::vector<cv::Mat>& hf_refs) {
    return cv::Mat();
}

cv::Mat sl_opencv::depth_compute(const std::vector<cv::Mat>& lf_objs,
                                 const std::vector<cv::Mat>& hf_objs) {
    return cv::Mat();
}
