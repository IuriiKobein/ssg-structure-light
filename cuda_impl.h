#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudev/common.hpp>

void invertArray(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y, cv::cuda::GpuMat &z);
void cudaSin(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y);
void cudaCos(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y);
void cudaRound(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y);