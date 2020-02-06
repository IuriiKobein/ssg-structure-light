#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudev/common.hpp>
#include <vector>

void cuda_invert_array(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y,
                       cv::cuda::GpuMat &z);

void cuda_sin(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y);
void cuda_cos(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y);
void cuda_round(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y);

void cuda_sin_cos(const cv::cuda::GpuMat &in, cv::cuda::GpuMat &sin_out,
                  cv::cuda::GpuMat &cos_out);

void cuda_diff_atan_inplace(const std::vector<cv::cuda::GpuMat> &input,
                            cv::cuda::GpuMat &out);

void cuda_dft2dct_out_convert(const cv::cuda::GpuMat &input,
                              const cv::cuda::GpuMat &dct_coeff,
                              cv::cuda::GpuMat &out);

void cuda_idft2idct_in_convert(const cv::cuda::GpuMat &d_input,
                               const cv::cuda::GpuMat &d_idct_coeff,
                               cv::cuda::GpuMat &d_out);

void cuda_delta_phi_mult_sub_inplace(cv::cuda::GpuMat &d_in1,
                                     const cv::cuda::GpuMat &d_in2,
                                     const cv::cuda::GpuMat &d_cos_f,
                                     const cv::cuda::GpuMat &d_sin_f);

void cuda_wrap(cv::cuda::GpuMat &x);
void cuda_temporal_unwrap(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y, 
                          cv::cuda::GpuMat &z, float scale);