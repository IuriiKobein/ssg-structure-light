#include "cu_alg_utils.hpp"
#include "cuda_kernels.h"

#include <cstdint>
#include <experimental/filesystem>
#include <iostream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

namespace {

std::vector<cv::cuda::GpuMat> cuda_imgs_from_dir_load_impl(
    const char *dir_path) {
    using namespace std::experimental;
    std::vector<std::string> path_list;

    for (const auto &entry : filesystem::directory_iterator(dir_path)) {
        path_list.emplace_back(entry.path());
    }

    std::sort(std::begin(path_list), std::end(path_list));

    for (const auto &p : path_list) {
        std::cout << p << '\n';
    }

    return cuda_imgs_load(path_list);
}

template <typename MatType>
static void to_float_scale(const std::vector<MatType> &src,
                           std::vector<MatType> &dst) {
    for (auto i = 0ul; i < src.size(); ++i) {
        src[i].convertTo(dst[i], CV_32FC1, 1.0f / 255);
    }
}

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
        [](const auto &p) { return cv::imread(p, cv::IMREAD_GRAYSCALE); });

    return imgs;
}
}  // namespace

void cuda_img_show(const std::string title, cv::cuda::GpuMat &d_img) {
    cv::Mat h_img(d_img.size(), d_img.type());

    d_img.download(h_img);
    img_show(title, h_img);
    cv::waitKey(0);
}

cv::cuda::GpuMat cuda_img_alloc(cv::Size size, int type) {
    return cv::cuda::GpuMat(size, type);
}

std::vector<cv::cuda::GpuMat> cuda_imgs_alloc(std::size_t num, cv::Size size,
                                              int type) {
    std::vector<cv::cuda::GpuMat> d_imgs;
    d_imgs.reserve(num);

    for (auto i = 0ul; i < num; ++i) {
        d_imgs.emplace_back(size, type);
    }

    return d_imgs;
}

std::vector<cv::cuda::GpuMat> cuda_imgs_load(
    const std::vector<std::string> &path_list) {
    auto h_imgs = host_imgs_load(path_list);
    auto d_imgs = cuda_imgs_alloc(h_imgs.size(), h_imgs[0].size(), CV_8U);

    for (auto i = 0ul; i < h_imgs.size(); ++i) {
        d_imgs[i].upload(h_imgs[i]);
    }

    return d_imgs;
}

std::vector<cv::cuda::GpuMat> cuda_imgs_from_dir_load(const char *dir_path) {
    return cuda_imgs_from_dir_load_impl(dir_path);
}

std::vector<cv::cuda::GpuMat> cuda_imgs_from_dir_load(
    const std::string &dir_path) {
    return cuda_imgs_from_dir_load_impl(dir_path.c_str());
}

void to_gpu_mat(const std::vector<cv::Mat> &src,
                std::vector<cv::cuda::GpuMat> &dst) {
    for (auto i = 0ul; i < src.size(); ++i) {
        dst[i].upload(src[i]);
    }
}

void cuda_phase_compute(const std::vector<cv::Mat> &src,
                        std::vector<cv::Mat> &tmp,
                        std::vector<cv::cuda::GpuMat> &cu_tmp,
                        cv::cuda::GpuMat &out, cv::cuda::Filter &filt) {
    to_float_scale<cv::Mat>(src, tmp);
    to_gpu_mat(tmp, cu_tmp);

    std::for_each(std::begin(cu_tmp), std::end(cu_tmp),
                  [&filt](auto &img) { filt.apply(img, img); });
    cuda_diff_atan_inplace(cu_tmp, out);
}
