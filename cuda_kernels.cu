#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

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
    const int ii = 2 * i;

    z(j, ii) = x(j, i);
    z(j, ii + 1) = y(j, (ii + 1) / 2);
}

__global__ void sin(cv::cuda::PtrStepSzf x, cv::cuda::PtrStepSzf y) {
    const int i = threadIdx.y;
    const int j = blockIdx.y;

    y(j, i) = sin(x(j, i));
}

__global__ void cos(cv::cuda::PtrStepSzf x, cv::cuda::PtrStepSzf y) {
    const int i = threadIdx.y;
    const int j = blockIdx.y;

    y(j, i) = cos(x(j, i));
}

__global__ void round(cv::cuda::PtrStepSzf x, cv::cuda::PtrStepSzf y) {
    const int i = threadIdx.y;
    const int j = blockIdx.y;

    y(j, i) = rint(x(j, i));
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
                                         cv::cuda::PtrStepSzf in1,
                                         const cv::cuda::PtrStepSzf cos_coeff,
                                         const cv::cuda::PtrStepSzf sin_coeff,
                                         cv::cuda::PtrStepSzf out) {
    const int x = threadIdx.y;
    const int y = blockIdx.y;

    out(y, x) = in0(y, x) * cos_coeff(y, x) - in1(y, x) * sin_coeff(y, x);
}

static cufftHandle plan;
void fft_2d_init(int h, int w) { cufftPlan2d(&plan, w * 2, h * 2, CUFFT_R2C); }

void fft_2d_exe(cv::cuda::GpuMat &in, cv::cuda::GpuMat &out) {
    cufftExecR2C(plan, in.ptr<cufftReal>(), out.ptr<cufftComplex>());
}

void invertArray(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y,
                 cv::cuda::GpuMat &z) {
    const dim3 block(1, 256);
    const dim3 grid(1, 512);

    invert2D<<<grid, block>>>(x, y, z);
}

void cudaSin(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y) {
    const dim3 block(1, 512);
    const dim3 grid(1, 512);
    sin<<<grid, block>>>(x, y);
}

void cudaCos(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y) {
    const dim3 block(1, 512);
    const dim3 grid(1, 512);

    cos<<<grid, block>>>(x, y);
}

void cudaRound(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y) {
    const dim3 block(1, 512);
    const dim3 grid(1, 512);

    round<<<grid, block>>>(x, y);
}

cv::cuda::GpuMat &cuda_diff_atan_inplace(
    std::vector<cv::cuda::GpuMat> &d_input) {
    const dim3 block(1, 512);
    const dim3 grid(1, 512);

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
    const dim3 block(1, 512);
    const dim3 grid(1, 512);

    delta_phi_inplace<<<grid, block>>>(d_in1, d_in2, d_cos_f, d_sin_f, d_out);
}

void cuda_sin_cos(const cv::cuda::GpuMat &in, cv::cuda::GpuMat &sin_out,
                  cv::cuda::GpuMat &cos_out) {
    const dim3 block(1, 512);
    const dim3 grid(1, 512);

    sin_cos<<<grid, block>>>(in, sin_out, cos_out);
}
