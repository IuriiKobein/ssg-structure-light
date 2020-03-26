#pragma once

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <string>

void img_show(const std::string title, cv::Mat &h_img);
void cuda_img_show(const std::string title, cv::cuda::GpuMat &d_img);
std::vector<cv::Mat> host_imgs_load(const std::vector<std::string> &path_list);
std::vector<cv::Mat> imgs_alloc(std::size_t num, cv::Size size, int type);
std::vector<cv::Mat> imgs_from_dir_load(const std::string &dir_path);
void cpu_phase_compute(const std::vector<cv::Mat> &src, cv::Mat &out);
void cv_round(cv::Mat &img);
