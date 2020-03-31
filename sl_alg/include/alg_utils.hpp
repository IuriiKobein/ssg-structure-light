#pragma once

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <string>

struct sinusoidal_pattern_params {
    cv::Size size;
    std::int32_t num_of_patterns;
    std::int32_t is_horizontal;
    std::int32_t num_of_periods;
};

// Generate sinusoidal patterns. Markers are optional
std::vector<cv::Mat> sinusoidal_pattern_generate(
    sinusoidal_pattern_params &params);

void lfs_img_write(const std::string &path, const cv::Mat &img);
void lfs_imgs_write(const std::string &path, const std::vector<cv::Mat> &imgs);

void img_show(const std::string &title, const cv::Mat &img);
void cuda_img_show(const std::string &title, cv::cuda::GpuMat &d_img);
std::vector<cv::Mat> host_imgs_load(const std::vector<std::string> &path_list);
std::vector<cv::Mat> imgs_alloc(std::size_t num, cv::Size size, int type);
std::vector<cv::Mat> imgs_from_dir_load(const std::string &dir_path);
void cpu_phase_compute(const std::vector<cv::Mat> &src, cv::Mat &out);
void cv_round(cv::Mat &img);
