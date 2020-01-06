#pragma once 

#include <vector> 
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudev/common.hpp>

void invertArray(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y, cv::cuda::GpuMat &z);
void cudaSin(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y);
void cudaCos(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y);
void cudaRound(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y);

cv::cuda::GpuMat& cuda_diff_atan_inplace(std::vector<cv::cuda::GpuMat> &d_input);
void cuda_dft2dct_out_convert(const cv::cuda::GpuMat &d_input,
                            const cv::cuda::GpuMat &d_cos_f,
                            const cv::cuda::GpuMat &d_sin_f, cv::cuda::GpuMat &d_out);

void cuda_idft2idct_in_convert(const cv::cuda::GpuMat &d_input,
                            const cv::cuda::GpuMat &d_cos_f,
                            const cv::cuda::GpuMat &d_sin_f, cv::cuda::GpuMat &d_out);

void cuda_delta_phi_mult_sub_inplace(cv::cuda::GpuMat &d_in1, cv::cuda::GpuMat &d_in2,
                            const cv::cuda::GpuMat &d_cos_f,
                            const cv::cuda::GpuMat &d_sin_f, cv::cuda::GpuMat &d_out);
