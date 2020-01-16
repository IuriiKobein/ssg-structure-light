#include <opencv2/core/cuda.hpp>
#include <vector>

struct ConstData
{   
    int height;
    int width;
    cv::cuda::GpuMat dct_twiddle;
    cv::cuda::GpuMat idct_twiddle;
    cv::cuda::GpuMat cudaGridLaplacian;
};

struct VarMats
{
    cv::cuda::GpuMat doubledMat;
    cv::cuda::GpuMat Mat;
    std::vector<cv::cuda::GpuMat> c_arr;
    cv::cuda::GpuMat fftOut;
    cv::cuda::GpuMat imgSin;
    cv::cuda::GpuMat imgCos;
    cv::cuda::GpuMat ca;
    cv::cuda::GpuMat a1;
    cv::cuda::GpuMat a2;    
    cv::cuda::GpuMat k1;
    cv::cuda::GpuMat k2;
    cv::cuda::GpuMat phi1;
    cv::cuda::GpuMat phi2;
    cv::cuda::GpuMat error;
    cv::cuda::GpuMat ifftIn;
    cv::cuda::GpuMat x;
};
