#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>
#include "cuda_impl.h"
#include <chrono>
#include <opencv2/imgcodecs.hpp>
#include <string> 

void phaseUnwrap(cv::cuda::GpuMat &img, cv::cuda::GpuMat &cudaCosDCT, cv::cuda::GpuMat &cudaSinDCT, 
                cv::cuda::GpuMat &cudaCosIDCT, cv::cuda::GpuMat &cudaSinIDCT, cv::cuda::GpuMat &cudaGridLaplacian);