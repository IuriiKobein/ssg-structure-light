#pragma once

#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>

void img_show(const std::string title, cv::Mat &h_img);
void cuda_img_show(const std::string title, cv::cuda::GpuMat &d_img);

std::vector<cv::Mat> host_imgs_load(const std::vector<std::string> &path_list);
std::vector<cv::cuda::GpuMat> cuda_imgs_alloc(std::size_t num, cv::Size size,
                                              int type);
std::vector<cv::cuda::GpuMat> cuda_imgs_load(
    const std::vector<std::string> &path_list);

std::vector<cv::cuda::GpuMat> cuda_imgs_from_dir_load(const char *dir_path);

std::vector<cv::cuda::GpuMat> cuda_imgs_alloc_type(int flag, cv::Size size,
                                              int type);
