#include "alg_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <experimental/filesystem>
#include <iostream>
#include <iterator>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

namespace {

std::vector<cv::Mat> imgs_from_dir_load_impl(const char *dir_path) {
    using namespace std::experimental;
    std::vector<std::string> path_list;

    for (const auto &entry : filesystem::directory_iterator(dir_path)) {
        path_list.emplace_back(entry.path());
    }

    std::sort(std::begin(path_list), std::end(path_list));

    for (const auto &p : path_list) {
        std::cout << p << '\n';
    }

    return host_imgs_load(path_list);
}

static void cpu_diff_atan(const std::vector<cv::Mat> &src, cv::Mat &dst) {
    for (int i = 0; i < src[0].rows; i++) {
        for (int j = 0; j < src[0].cols; j++) {
            float x = src[3].at<float>(i, j) - src[1].at<float>(i, j);
            float y = src[0].at<float>(i, j) - src[2].at<float>(i, j);
            dst.at<float>(i, j) = std::atan2(x, y);
        }
    }
}

template <typename MatType>
static void to_float_scale(const std::vector<MatType> &src,
                           std::vector<MatType> &dst) {
    for (auto i = 0ul; i < src.size(); ++i) {
        src[i].convertTo(dst[i], CV_32FC1, 1.0f / 255);
    }
}
}  // namespace

void img_show(const std::string &title, const cv::Mat &img) {
    auto tmp = img.clone();

    cv::normalize(tmp, tmp, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::namedWindow(title, cv::WINDOW_NORMAL);
    cv::imshow(title, tmp);
}

void lfs_img_write(const std::string &path, const cv::Mat &h_img) {
    cv::normalize(h_img, h_img, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imwrite(path, h_img);
}

void lfs_imgs_write(const std::string &path, const  std::vector<cv::Mat> &imgs) {
    std::int32_t i = 0;

    std::for_each(std::begin(imgs), std::end(imgs),
                  [&i, &path](const auto &img) {
                      lfs_img_write(std::string(path) + std::to_string(i++) + ".png", img);
                  });
}

std::vector<cv::Mat> host_imgs_load(const std::vector<std::string> &path_list) {
    std::vector<cv::Mat> imgs;
    imgs.reserve(path_list.size());

    std::transform(
        std::begin(path_list), std::end(path_list), std::back_inserter(imgs),
        [](const auto &p) { return cv::imread(p, cv::IMREAD_GRAYSCALE); });

    return imgs;
}

std::vector<cv::Mat> imgs_alloc(std::size_t num, cv::Size size, int type) {
    std::vector<cv::Mat> h_imgs;
    h_imgs.reserve(num);

    for (auto i = 0ul; i < num; ++i) {
        h_imgs.emplace_back(size, type);
    }

    return h_imgs;
}

std::vector<cv::Mat> imgs_from_dir_load(const std::string &dir_path) {
    return imgs_from_dir_load_impl(dir_path.c_str());
}

void cpu_phase_compute(const std::vector<cv::Mat> &src, cv::Mat &out) {
    std::vector<cv::Mat> tmp(src.size());

    to_float_scale<cv::Mat>(src, tmp);
    std::for_each(std::begin(tmp), std::end(tmp), [](auto &img) {
        cv::GaussianBlur(img, img, cv::Size(5, 5), 0);
    });
    cpu_diff_atan(tmp, out);
}

void cv_round(cv::Mat &img) {
    for (auto i = 0; i < img.rows; i++) {
        for (auto j = 0; j < img.cols; j++) {
            img.at<float>(i, j) = round(img.at<float>(i, j));
        }
    }
}

// Generate sinusoidal patterns. Markers are optional
std::vector<cv::Mat> sinusoidal_pattern_generate(
    sinusoidal_pattern_params &params) {
    // Three patterns are used in the reference paper.
    std::float_t meanAmpl = 127.5;
    std::float_t sinAmpl = 127.5;
    std::float_t phase_shift = std::float_t(2 * CV_PI / params.num_of_patterns);
    std::int32_t period_pixels;
    std::float_t freq;

    std::vector<cv::Mat> patterns(params.num_of_patterns);

    if (params.is_horizontal) {
        period_pixels = params.size.height / params.num_of_periods;
    } else {
        period_pixels = params.size.width / params.num_of_periods;
    }

    freq = (std::float_t)(1 / period_pixels);

    for (int i = 0; i < params.num_of_patterns; ++i) {
        patterns[i] = cv::Mat(params.size.height, params.size.width, CV_8UC1);

        if (params.is_horizontal) {
            patterns[i] = patterns[i].t();
        }
    }
    // Patterns vary along one direction only so, a row Mat can be created and
    // copied to the pattern's rows
    for (int i = 0; i < params.num_of_patterns; ++i) {
        cv::Mat rowValues(1, patterns[i].cols, CV_8UC1);

        for (int j = 0; j < patterns[i].cols; ++j) {
            rowValues.at<uchar>(0, j) = cv::saturate_cast<uchar>(
                meanAmpl +
                sinAmpl * sin(2 * CV_PI * freq * j + i * phase_shift));
        }

        for (int j = 0; j < patterns[i].rows; ++j) {
            rowValues.row(0).copyTo(patterns[i].row(j));
        }
    }

    if (params.is_horizontal) {
        for (int i = 0; i < params.num_of_patterns; ++i) {
            patterns[i] = patterns[i].t();
        }
    }

    return patterns;
}

