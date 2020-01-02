#include <opencv2/core/cuda.hpp>
#include <vector>

struct ConstData
{   
    int height;
    int width;
    cv::cuda::GpuMat cudaCosDCT;
    cv::cuda::GpuMat cudaSinDCT;
    cv::cuda::GpuMat cudaCosIDCT;
    cv::cuda::GpuMat cudaSinIDCT;
    cv::cuda::GpuMat cudaGridLaplacian;
};

struct VarMats
{
    cv::cuda::GpuMat doubledMat;
    cv::cuda::GpuMat Mat;
    cv::cuda::GpuMat complexMat;
    cv::cuda::GpuMat outMat;
    std::vector<cv::cuda::GpuMat> complexArray;
    cv::cuda::GpuMat ch1;
    cv::cuda::GpuMat ch2;
    cv::cuda::GpuMat fftOut;
    cv::cuda::GpuMat imgSin;
    cv::cuda::GpuMat imgCos;
    cv::cuda::GpuMat ca;
    cv::cuda::GpuMat ica;
    cv::cuda::GpuMat a1;
    cv::cuda::GpuMat a2;    
    cv::cuda::GpuMat k1;
    cv::cuda::GpuMat k1round;
    cv::cuda::GpuMat k2;
    cv::cuda::GpuMat k2round;
    cv::cuda::GpuMat phi1;
    cv::cuda::GpuMat phi2;
    cv::cuda::GpuMat error;
    cv::cuda::GpuMat ifftIn;
    cv::cuda::GpuMat z;
    cv::cuda::GpuMat x;
};