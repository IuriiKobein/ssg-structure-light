#pragma once

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/highgui.hpp>
#include <string>

void cuda_img_show(const std::string title, cv::cuda::GpuMat &d_img);

cv::cuda::GpuMat cuda_img_alloc(cv::Size size, int type);
std::vector<cv::cuda::GpuMat> cuda_imgs_alloc(std::size_t num, cv::Size size,
                                              int type);
std::vector<cv::cuda::GpuMat> cuda_imgs_load(
    const std::vector<std::string> &path_list);

std::vector<cv::cuda::GpuMat> cuda_imgs_from_dir_load(const char *dir_path);
std::vector<cv::cuda::GpuMat> cuda_imgs_from_dir_load(
    const std::string &dir_path);

void cuda_phase_compute(const std::vector<cv::Mat> &src,
                        std::vector<cv::Mat> &tmp,
                        std::vector<cv::cuda::GpuMat> &cu_tmp,
                        cv::cuda::GpuMat &out, cv::cuda::Filter &filt);
