#include "cuda_impl.h"
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void invert(cv::cuda::PtrStepSzf x, cv::cuda::PtrStepSzf y,
                       cv::cuda::PtrStepSzf z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    if (i % 2 == 0)
        z(0, i) = x(0, i / 2);
    else
        z(0, i) = y(0, (i - 1) / 2);
}

__global__ void invert2D(cv::cuda::PtrStepSzf x, cv::cuda::PtrStepSzf y,
                         cv::cuda::PtrStepSzf z) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= x.cols || j >= x.rows) return;

    if (j % 2 == 0)
        z(i, j) = x(i, j / 2);
    else
        z(i, j) = y(i, (j - 1) / 2);
}

__global__ void sin(cv::cuda::PtrStepSzf x, cv::cuda::PtrStepSzf y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= x.cols || j >= x.rows) return;

    y(i, j) = sin(x(i, j));
}

__global__ void cos(cv::cuda::PtrStepSzf x, cv::cuda::PtrStepSzf y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= x.cols || j >= x.rows) return;

    y(i, j) = cos(x(i, j));
}

__global__ void round(cv::cuda::PtrStepSzf x, cv::cuda::PtrStepSzf y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= x.cols || j >= x.rows) return;

    y(i, j) = rint(x(i, j));
}

static __global__ void img_atan_inplace(cv::cuda::PtrStepSzf in0,
                                        cv::cuda::PtrStepSzf in1,
                                        cv::cuda::PtrStepSzf in2,
                                        cv::cuda::PtrStepSzf in3,
                                        cv::cuda::PtrStepSzf out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= in0.cols || y >= in0.rows) return;

    out(x, y) = atan2f(in3(x, y) - in1(x, y), in0(x, y) - in2(x, y));
}

void invertArray(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y,
                 cv::cuda::GpuMat &z) {
    const dim3 block(16, 16);
    const dim3 grid(cv::cudev::divUp(x.cols, block.x),
                    cv::cudev::divUp(x.rows, block.y));
    invert2D<<<grid, block>>>(x, y, z);
}

void cudaSin(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y) {
    const dim3 block(16, 16);
    const dim3 grid(cv::cudev::divUp(x.cols, block.x),
                    cv::cudev::divUp(x.rows, block.y));
    sin<<<grid, block>>>(x, y);
}

void cudaCos(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y) {
    const dim3 block(16, 16);
    const dim3 grid(cv::cudev::divUp(x.cols, block.x),
                    cv::cudev::divUp(x.rows, block.y));
    cos<<<grid, block>>>(x, y);
}

void cudaRound(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y) {
    const dim3 block(16, 16);
    const dim3 grid(cv::cudev::divUp(x.cols, block.x),
                    cv::cudev::divUp(x.rows, block.y));
    round<<<grid, block>>>(x, y);
}

cv::cuda::GpuMat& cuda_diff_atan_inplace(std::vector<cv::cuda::GpuMat> &d_input) {
    const dim3 block(16, 16);

    const dim3 grid(cv::cudev::divUp(d_input[0].cols, block.x),
                    cv::cudev::divUp(d_input[0].rows, block.y));

    img_atan_inplace<<<grid, block>>>(d_input[0], d_input[1], d_input[2],
                                      d_input[3], d_input[0]);

    return d_input[0];
}
