#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include "cuda_kernels.h"

__global__ void invert(cv::cuda::PtrStepSzf x, cv::cuda::PtrStepSzf y,
                       cv::cuda::PtrStepSzf z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i % 2 == 0)
        z(0, i) = x(0, i / 2);
    else
        z(0, i) = y(0, (i - 1) / 2);
}

__global__ void invert2D(cv::cuda::PtrStepSzf x, cv::cuda::PtrStepSzf y,
                         cv::cuda::PtrStepSzf z) {
    const int i = threadIdx.y;
    const int j = blockIdx.y;

    if (j % 2 == 0) {
        z(i, j) = x(i, j/2);
        //    z(4*i + 1, 4*j + 1) = x(4*i + 1, (4*j + 1) / 2);
        //    z(4*i + 2, 4*j + 2) = x(4*i + 2, 2*j + 1);
        //    z(4*i + 3, 4*j + 3) = x(4*i + 3, (4*j + 3) / 2);
    } else {
        z(i, j) = y(i, (j - 1) / 2);
        //    z(4*i + 1, 4*j + 1) = y(4*i + 1, 2*j);
        //    z(4*i + 2, 4*j + 2) = y(4*i + 2, (4*j + 1)/2);
        //    z(4*i + 3, 4*j + 3) = y(4*i + 3, 2*j + 1);
    }
}

__global__ void invert2D_even(cv::cuda::PtrStepSzf x, cv::cuda::PtrStepSzf z) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int ii = 2 * i;
    const int jj = 2 * j;

    z(ii, jj) = x(ii, j);
    z(ii + 1, jj + 1) = x(ii + 1, (2 * j + 1) / 2);
}

__global__ void invert2D_odd(cv::cuda::PtrStepSzf y, cv::cuda::PtrStepSzf z) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int ii = 2 * i;
    const int jj = 2 * j;

    z(ii, jj) = y(ii, j - 1);
    z(ii + 1, jj + 1) = y(ii, j);
}

__global__ void sin(cv::cuda::PtrStepSzf x, cv::cuda::PtrStepSzf y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    y(i, j) = sin(x(i, j));
}

__global__ void cos(cv::cuda::PtrStepSzf x, cv::cuda::PtrStepSzf y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    y(i, j) = cos(x(i, j));
}

__global__ void round(cv::cuda::PtrStepSzf x, cv::cuda::PtrStepSzf y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    y(i, j) = rint(x(i, j));
}

static __global__ void img_atan_inplace(cv::cuda::PtrStepSzf in0,
                                        cv::cuda::PtrStepSzf in1,
                                        cv::cuda::PtrStepSzf in2,
                                        cv::cuda::PtrStepSzf in3,
                                        cv::cuda::PtrStepSzf out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    out(x, y) = atan2f(in3(x, y) - in1(x, y), in0(x, y) - in2(x, y));
}

static __global__ void dft2dct_inplace(cv::cuda::PtrStepSz<float2> in0,
                                       cv::cuda::PtrStepSzf cos_coeff,
                                       cv::cuda::PtrStepSzf sin_coeff,
                                       cv::cuda::PtrStepSzf out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    out(x, y) = in0(x, y).x * cos_coeff(x, y) + in0(x, y).y * sin_coeff(x, y);
}

static __global__ void vec_comp_elem_wise_mul2(
    cv::cuda::PtrStepSz<float2> in0, cv::cuda::PtrStepSz<float2> dct_coeff,
    cv::cuda::PtrStepSzf out) {
    const int x = threadIdx.y;
    const int y = blockIdx.y;

    float2 in = in0(y, x);
    float2 coeff = dct_coeff(y, x);

    out(y, x) = in.x * coeff.x + in.y * coeff.y;
}

static __global__ void vec_real_elem_wise_mul2(
    cv::cuda::PtrStepSzf in, cv::cuda::PtrStepSz<float2> idct_coeff,
    cv::cuda::PtrStepSz<float2> out) {
    const int x = threadIdx.y;
    const int y = blockIdx.y;

    float in0 = in(y, x);
    float2 coeff0 = idct_coeff(y, x);
    float2 r0;

    r0.x = in0 * coeff0.x;
    r0.y = in0 * coeff0.y;

    out(y, x) = r0;
}

static __global__ void delta_phi_inplace(cv::cuda::PtrStepSzf in0,
                                         cv::cuda::PtrStepSzf in1,
                                         cv::cuda::PtrStepSzf cos_coeff,
                                         cv::cuda::PtrStepSzf sin_coeff,
                                         cv::cuda::PtrStepSzf out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int bx = threadIdx.x;
    const int by = threadIdx.y;

    __shared__ float sin0[16][16];
    __shared__ float sin1[16][16];
    __shared__ float sin[16][16];
    __shared__ float cos[16][16];

    sin0[bx][by] = in0(x, y);
    sin1[bx][by] = in1(x, y);
    sin[bx][by] = sin_coeff(x, y);
    cos[bx][by] = cos_coeff(x, y);

    __syncthreads();

    out(x, y) = sin0[bx][by] * cos[bx][by] - sin1[bx][by] * sin[bx][by];
}

void invertArray(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y,
                 cv::cuda::GpuMat &z) {
    const dim3 block(1, 512);
    const dim3 grid(1, 512);

    invert2D<<<grid, block>>>(x, y, z);
}

void invertArray2(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y,
                  cv::cuda::GpuMat &z) {
    const dim3 block(16, 16);
    const dim3 grid(cv::cudev::divUp(x.cols / 2, block.x),
                    cv::cudev::divUp(x.rows / 2, block.y));
    invert2D_even<<<grid, block>>>(x, z);
    invert2D_odd<<<grid, block>>>(y, z);
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

cv::cuda::GpuMat &cuda_diff_atan_inplace(
    std::vector<cv::cuda::GpuMat> &d_input) {
    const dim3 block(16, 16);
    const dim3 grid(cv::cudev::divUp(d_input[0].cols, block.x),
                    cv::cudev::divUp(d_input[0].rows, block.y));

    img_atan_inplace<<<grid, block>>>(d_input[0], d_input[1], d_input[2],
                                      d_input[3], d_input[0]);

    return d_input[0];
}

void cuda_dft2dct_out_convert(const cv::cuda::GpuMat &d_input,
                              const cv::cuda::GpuMat &d_dct_coeff,
                              cv::cuda::GpuMat &d_out) {
    const dim3 block(1, 512);
    const dim3 grid(1, 512);

    vec_comp_elem_wise_mul2<<<grid, block>>>(d_input, d_dct_coeff,
                                             d_out.reshape(1));
}

void cuda_idft2idct_in_convert(const cv::cuda::GpuMat &d_input,
                               const cv::cuda::GpuMat &d_idct_coeff,
                               cv::cuda::GpuMat &d_out) {
    const dim3 block(1, 512);
    const dim3 grid(1, 512);

    vec_real_elem_wise_mul2<<<grid, block>>>(d_input, d_idct_coeff, d_out);
}

void cuda_delta_phi_mult_sub_inplace(cv::cuda::GpuMat &d_in1,
                                     cv::cuda::GpuMat &d_in2,
                                     const cv::cuda::GpuMat &d_cos_f,
                                     const cv::cuda::GpuMat &d_sin_f,
                                     cv::cuda::GpuMat &d_out) {
    const dim3 block(16, 16);

    const dim3 grid(cv::cudev::divUp(d_in1.cols, block.x),
                    cv::cudev::divUp(d_in1.rows, block.y));

    delta_phi_inplace<<<grid, block>>>(d_in1, d_in2, d_cos_f, d_sin_f, d_out);
}
