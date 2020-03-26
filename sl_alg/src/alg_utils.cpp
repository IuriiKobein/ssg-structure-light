#include "alg_utils.hpp"

#include <cstdint>
#include <experimental/filesystem>
#include <iostream>
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

void img_show(const std::string title, cv::Mat &h_img) {
    cv::normalize(h_img, h_img, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::namedWindow(title, cv::WINDOW_NORMAL);
    cv::imshow(title, h_img);
}

std::vector<cv::Mat> host_imgs_load(const std::vector<std::string> &path_list) {
    std::vector<cv::Mat> imgs;
    imgs.reserve(path_list.size());

    std::transform(
        std::begin(path_list), std::end(path_list), std::back_inserter(imgs),
        [](const auto &p) {
            return cv::imread(p, cv::IMREAD_GRAYSCALE);
        });

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
