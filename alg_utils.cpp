#include "alg_utils.hpp"

#include <experimental/filesystem>
#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgcodecs.hpp>

void img_show(const std::string title, cv::Mat &h_img) {
    cv::normalize(h_img, h_img, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::namedWindow(title, cv::WINDOW_NORMAL);
    cv::imshow(title, h_img);
}

void cuda_img_show(const std::string title, cv::cuda::GpuMat &d_img) {
    cv::Mat h_img(d_img.size(), d_img.type());

    d_img.download(h_img);
    img_show(title, h_img);
}

std::vector<cv::Mat> host_imgs_load(const std::vector<std::string> &path_list) {
    std::vector<cv::Mat> imgs;
    imgs.reserve(path_list.size());

    std::transform(
        std::begin(path_list), std::end(path_list), std::back_inserter(imgs),
        [](const auto &p) { return cv::imread(p, cv::IMREAD_GRAYSCALE); });

    return imgs;
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

std::vector<cv::cuda::GpuMat> cuda_imgs_alloc_type(int flag, cv::Size size,
                                              int type) {
    std::vector<cv::cuda::GpuMat> d_imgs;

    d_imgs.reserve(flag);

    for (auto i = 0ul; i < flag; ++i) {
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


