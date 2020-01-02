#include "cuda_impl.h"

__global__ void invert(cv::cuda::PtrStepSzf x, cv::cuda::PtrStepSzf y, cv::cuda::PtrStepSzf  z, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (i >= n) return;
  
  if(i%2==0) z(0, i) = x(0, i/2);
  else z(0, i) = y(0, (i-1)/2);
}

__global__ void invert2D(cv::cuda::PtrStepSzf x, cv::cuda::PtrStepSzf y, cv::cuda::PtrStepSzf z){
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= x.cols || j >= x.rows) return;
    
  if(j%2==0) z(i, j) = x(i, j/2);
  else z(i, j) = y(i, (j-1)/2);
}

__global__ void sin(cv::cuda::PtrStepSzf x, cv::cuda::PtrStepSzf y){
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= x.cols || j >= x.rows) return;
  
  y(i, j) = sin(x(i,j));
}

__global__ void cos(cv::cuda::PtrStepSzf x, cv::cuda::PtrStepSzf y){
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= x.cols || j >= x.rows) return;
  
  y(i, j) = cos(x(i,j));
}

__global__ void round(cv::cuda::PtrStepSzf x, cv::cuda::PtrStepSzf y){
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= x.cols || j >= x.rows) return;
  
  y(i, j) = rint(x(i,j));
}

void invertArray(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y, cv::cuda::GpuMat &z){   
  const dim3 block(16, 16);
  const dim3 grid(cv::cudev::divUp(x.cols, block.x),
                  cv::cudev::divUp(x.rows, block.y));
  invert2D<<<grid, block>>>(x, y, z);
}

void cudaSin(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y){   
  const dim3 block(16, 16);
  const dim3 grid(cv::cudev::divUp(x.cols, block.x),
                  cv::cudev::divUp(x.rows, block.y));
  sin<<<grid, block>>>(x, y);
}

void cudaCos(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y){   
  const dim3 block(16, 16);
  const dim3 grid(cv::cudev::divUp(x.cols, block.x),
                  cv::cudev::divUp(x.rows, block.y));
  cos<<<grid, block>>>(x, y);
}

void cudaRound(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y){   
  const dim3 block(16, 16);
  const dim3 grid(cv::cudev::divUp(x.cols, block.x),
                  cv::cudev::divUp(x.rows, block.y));
  round<<<grid, block>>>(x, y);
}