#include <cuda.h>
#include <cuda_runtime.h>

#include <opencv2/core/cuda.hpp>
#include "cuda_kernels.h"

static __global__ void invert2D(cv::cuda::PtrStepSzf x, cv::cuda::PtrStepSzf y,
                         cv::cuda::PtrStepSzf z) {
    const int i = threadIdx.y;
    const int j = blockIdx.y;
    const int ii = 2 * i;

    z(j, ii) = x(j, i);
    z(j, ii + 1) = y(j, (ii + 1) / 2);
}

static __global__ void sin(cv::cuda::PtrStepSzf x, cv::cuda::PtrStepSzf y) {
    const int i = threadIdx.y;
    const int j = blockIdx.y;

    y(j, i) = sin(x(j, i));
}

static __global__ void cos(cv::cuda::PtrStepSzf x, cv::cuda::PtrStepSzf y) {
    const int i = threadIdx.y;
    const int j = blockIdx.y;

    y(j, i) = cos(x(j, i));
}

static __constant__ float pre = 0.5 / M_PI;
static __constant__ float post = 2 * M_PI;

static __global__ void round(cv::cuda::PtrStepSzf x, cv::cuda::PtrStepSzf y) {
    const int i = threadIdx.y;
    const int j = blockIdx.y;

    y(j, i) = rint(x(j, i) * pre) * post;
}

static __global__ void sin_cos(const cv::cuda::PtrStepSzf in,
                               cv::cuda::PtrStepSzf sin_out,
                               cv::cuda::PtrStepSzf cos_out) {
    const int i = threadIdx.y;
    const int j = blockIdx.y;

    sin_out(j, i) = sin(in(j, i));
    cos_out(j, i) = cos(in(j, i));
}

static __global__ void img_atan_inplace(cv::cuda::PtrStepSzf in0,
                                        const cv::cuda::PtrStepSzf in1,
                                        const cv::cuda::PtrStepSzf in2,
                                        const cv::cuda::PtrStepSzf in3,
                                        cv::cuda::PtrStepSzf out) {
    const int x = threadIdx.y;
    const int y = blockIdx.y;

    out(y, x) = atan2f(in3(y, x) - in1(y, x), in0(y, x) - in2(y, x));
}

static __global__ void vec_comp_elem_wise_mul2(
    cv::cuda::PtrStepSz<float2> in0,
    const cv::cuda::PtrStepSz<float2> dct_coeff, cv::cuda::PtrStepSzf out) {
    const int x = threadIdx.y;
    const int y = blockIdx.y;

    float2 in = in0(y, x);
    float2 coeff = dct_coeff(y, x);

    float res = in.x * coeff.x + in.y * coeff.y;

    out(y, x) = res;
}

static __global__ void vec_real_elem_wise_mul2(
    cv::cuda::PtrStepSzf in, const cv::cuda::PtrStepSz<float2> idct_coeff,
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
                                         const cv::cuda::PtrStepSzf in1,
                                         const cv::cuda::PtrStepSzf cos_coeff,
                                         const cv::cuda::PtrStepSzf sin_coeff) {
    const int x = threadIdx.y;
    const int y = blockIdx.y;

    in0(y, x) = in0(y, x) * cos_coeff(y, x) - in1(y, x) * sin_coeff(y, x);
}

void cuda_invert_array(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y,
                 cv::cuda::GpuMat &z) {
    const dim3 block(1, x.rows / 2);
    const dim3 grid(1, x.cols);

    invert2D<<<grid, block>>>(x, y, z);
}

void cuda_sin(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y) {
    const dim3 block(1, x.rows);
    const dim3 grid(1, x.cols);

    sin<<<grid, block>>>(x, y);
}

void cuda_cos(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y) {
    const dim3 block(1, x.rows);
    const dim3 grid(1, x.cols);

    cos<<<grid, block>>>(x, y);
}

void cuda_round(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y) {
    const dim3 block(1, x.rows);
    const dim3 grid(1, x.cols);

    round<<<grid, block>>>(x, y);
}

void cuda_diff_atan_inplace(
    const std::vector<cv::cuda::GpuMat> &input,
    cv::cuda::GpuMat& out) {
    const dim3 block(1, 512);
    const dim3 grid(1, 512);

    img_atan_inplace<<<grid, block>>>(input[0], input[1], input[2],
                                      input[3], out);
}

void cuda_dft2dct_out_convert(const cv::cuda::GpuMat &d_input,
                              const cv::cuda::GpuMat &d_dct_coeff,
                              cv::cuda::GpuMat &d_out) {
    const dim3 block(1, d_input.rows);
    const dim3 grid(1, d_input.cols);

    vec_comp_elem_wise_mul2<<<grid, block>>>(d_input, d_dct_coeff,
                                             d_out.reshape(1));
}

void cuda_idft2idct_in_convert(const cv::cuda::GpuMat &d_input,
                               const cv::cuda::GpuMat &d_idct_coeff,
                               cv::cuda::GpuMat &d_out) {
    const dim3 block(1, d_input.rows);
    const dim3 grid(1, d_input.cols);

    vec_real_elem_wise_mul2<<<grid, block>>>(d_input, d_idct_coeff, d_out);
}

void cuda_delta_phi_mult_sub_inplace(cv::cuda::GpuMat &in1,
                                     const cv::cuda::GpuMat &in2,
                                     const cv::cuda::GpuMat &d_cos_f,
                                     const cv::cuda::GpuMat &d_sin_f) {
    const dim3 block(1, in1.rows);
    const dim3 grid(1, in1.cols);

    delta_phi_inplace<<<grid, block>>>(in1, in2, d_cos_f, d_sin_f);
}

void cuda_sin_cos(const cv::cuda::GpuMat &in, cv::cuda::GpuMat &sin_out,
                  cv::cuda::GpuMat &cos_out) {
    const dim3 block(1, in.rows);
    const dim3 grid(1, in.cols);

    sin_cos<<<grid, block>>>(in, sin_out, cos_out);
}
