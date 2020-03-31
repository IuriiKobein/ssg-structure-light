#include "sl_pcg.hpp"
#include "sl_alg_factory.hpp"

#include "alg_utils.hpp"

#include <algorithm>
#include <iterator>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>

#include <string>
#include <type_traits>
#include <vector>

namespace {

sl_alg_auto_reg s_sl_pcg_reg("cpu_pcg", [](const sl_alg::params_t& params) {
    return std::make_unique<sl_pcg>(params);
});

cv::Mat g_laplacian;

cv::Mat create_grid(cv::Size size) {
    auto gridX = cv::Mat(size, CV_32FC1);
    auto gridY = cv::Mat(size, CV_32FC1);

    for (auto i = 0; i < size.height; i++) {
        for (auto j = 0; j < size.width; j++) {
            gridX.at<float>(i, j) = j + 1;
            gridY.at<float>(i, j) = i + 1;
        }
    }
    cv::multiply(gridX, gridX, gridX);
    cv::multiply(gridY, gridY, gridY);

    cv::Mat output;
    cv::add(gridX, gridY, output);

    return output;
}

cv::Mat wrap_sin(cv::Mat &img) {
    auto imgSin = cv::Mat(img.rows, img.cols, CV_32FC1);

    for (auto i = 0; i < img.rows; i++) {
        for (auto j = 0; j < img.cols; j++) {
            float x = img.at<float>(i, j);
            while (abs(x) >= M_PI_2) {
                x += ((x > 0) - (x < 0)) * (M_PI - 2 * abs(x));
            }
            imgSin.at<float>(i, j) = x;
        }
    }

    return imgSin;
}

cv::Mat wrap_cos(cv::Mat &img) {
    auto imgCos = cv::Mat(img.rows, img.cols, CV_32FC1);

    for (auto i = 0; i < img.rows; i++) {
        for (auto j = 0; j < img.cols; j++) {
            float x = img.at<float>(i, j) - M_PI_2;
            while (abs(x) > M_PI_2) {
                x += ((x > 0) - (x < 0)) * (M_PI - 2 * abs(x));
            }
            imgCos.at<float>(i, j) = -x;
        }
    }

    return imgCos;
}

cv::Mat mat_sin(cv::Mat &img) {
    auto imgSin = cv::Mat(img.rows, img.cols, CV_32FC1);
    for (auto i = 0; i < img.rows; i++) {
        for (auto j = 0; j < img.cols; j++) {
            imgSin.at<float>(i, j) = sin(img.at<float>(i, j));
        }
    }
    return imgSin;
}

cv::Mat mat_cos(cv::Mat &img) {
    auto imgCos = cv::Mat(img.rows, img.cols, CV_32FC1);

    for (auto i = 0; i < img.rows; i++) {
        for (auto j = 0; j < img.cols; j++) {
            imgCos.at<float>(i, j) = cos(img.at<float>(i, j));
        }
    }

    return imgCos;
}

cv::Mat laplacian(cv::Mat &img, cv::Mat &grid) {
    cv::Mat output;
    cv::dct(img, output);
    cv::multiply(output, grid, output);
    cv::dct(output, output, cv::DCT_INVERSE);
    output = output * (-4 * M_PI * M_PI / (img.rows * img.cols));
    return output;
}

cv::Mat ilaplacian(cv::Mat &img, cv::Mat &grid) {
    cv::Mat output;

    cv::dct(img, output);
    cv::divide(output, grid, output);
    cv::dct(output, output, cv::DCT_INVERSE);
    output = output * (-img.rows * img.cols) / (4 * M_PI * M_PI);

    return output;
}

cv::Mat delta_compute(cv::Mat &img, cv::Mat &grid) {
    cv::Mat x1, x2;
    cv::Mat img_sin = wrap_sin(img);
    cv::Mat img_cos = wrap_cos(img);

    cv::multiply(img_cos, laplacian(img_sin, grid), x1);
    cv::multiply(img_sin, laplacian(img_cos, grid), x2);
    cv::subtract(x1, x2, x1);

    return x1;
}

cv::Mat pcg_phase_unwrap(cv::Mat &img) {
    cv::Mat phase1 = delta_compute(img, g_laplacian);
    cv::Mat error, k1, k2, phase2;

    phase1 = ilaplacian(phase1, g_laplacian);
    cv::subtract(phase1, img, k1);
    k1 *= 0.5 / M_PI;
    cv_round(k1);
    k1 *= 2 * M_PI;
    cv::add(img, k1, phase2);

    for (auto i = 0; i < 10; i++) {
        cv::subtract(phase2, phase1, error);
        cv::Mat phiError = delta_compute(error, g_laplacian);
        phiError = ilaplacian(phiError, g_laplacian);
        cv::add(phase1, phiError, phase1);
        cv::subtract(phase1, img, k2);
        k2 *= 0.5 / M_PI;
        cv_round(k2);
        k2 *= 2 * M_PI;
        cv::add(img, k2, phase2);
        k2.copyTo(k1);
    }

    return phase2;
}
}  // namespace

class sl_pcg::sl_pcg_impl {
   public:
    sl_pcg_impl(const params_t &params)
        : _params(params),
          _obj_phase(_params.size, CV_32F),
          _ref_phase(_params.size, CV_32F) {
        g_laplacian = create_grid(_params.size);
    }

    int ref_phase_compute(const std::vector<cv::Mat> &refs) {
        cpu_phase_compute(refs, _ref_phase);
        return 0;
    }

    cv::Mat depth_compute(const std::vector<cv::Mat> &objs) {
        cpu_phase_compute(objs, _obj_phase);
        cv::subtract(_obj_phase, _ref_phase, _obj_phase);

        return pcg_phase_unwrap(_obj_phase);
    }

   private:
    params_t _params;
    cv::Mat _obj_phase;
    cv::Mat _ref_phase;
};

sl_pcg::sl_pcg(const params_t &params)
    : _pimpl(std::make_unique<sl_pcg_impl>(params)) {}

sl_pcg::~sl_pcg() = default;

int sl_pcg::ref_phase_compute(const std::vector<cv::Mat> &ref_phases) {
    return _pimpl->ref_phase_compute(ref_phases);
}

cv::Mat sl_pcg::depth_compute(const std::vector<cv::Mat> &obj_phases) {
    return _pimpl->depth_compute(obj_phases);
}

int sl_pcg::ref_phase_compute(const std::vector<cv::Mat> &,
                              const std::vector<cv::Mat> &) {
    return -ENOTSUP;
}

cv::Mat sl_pcg::depth_compute(const std::vector<cv::Mat> &,
                              const std::vector<cv::Mat> &) {
    return cv::Mat();
}
