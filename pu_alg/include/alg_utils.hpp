#pragma once

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/highgui.hpp>
#include <string>

void img_show(const std::string title, cv::Mat &h_img);
void cuda_img_show(const std::string title, cv::cuda::GpuMat &d_img);

std::vector<cv::Mat> host_imgs_load(const std::vector<std::string> &path_list);

cv::cuda::GpuMat cuda_img_alloc(cv::Size size, int type);
std::vector<cv::cuda::GpuMat> cuda_imgs_alloc(std::size_t num, cv::Size size,
                                              int type);
std::vector<cv::cuda::GpuMat> cuda_imgs_load(
    const std::vector<std::string> &path_list);

std::vector<cv::cuda::GpuMat> cuda_imgs_from_dir_load(const char *dir_path);
std::vector<cv::cuda::GpuMat> cuda_imgs_from_dir_load(
    const std::string &dir_path);

void to_float_scale(const std::vector<cv::cuda::GpuMat> &src,
                    std::vector<cv::cuda::GpuMat> &dst);

void cuda_phase_compute(const std::vector<cv::cuda::GpuMat> &src,
                        std::vector<cv::cuda::GpuMat> &dst,
                        cv::cuda::GpuMat &out, cv::cuda::Filter &filt);
