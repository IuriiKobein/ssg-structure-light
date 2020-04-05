#include <cuda.h>
#include <cuda_runtime.h>

#include <opencv2/core/cuda.hpp>
#include "cuda_kernels.h"

static __global__ void invert2D(cv::cuda::PtrStepSzf in1, cv::cuda::PtrStepSzf in2,
                         cv::cuda::PtrStepSzf out) {
    const int x = blockIdx.y;
    const int y = threadIdx.y;
    const int yy = 2 * y;

    out(x, yy) = in1(x, y);
    out(x, yy + 1) = in2(x, (yy + 1) / 2);
}

static __global__ void sin(cv::cuda::PtrStepSzf in, cv::cuda::PtrStepSzf out) {
    const int x = blockIdx.y;
    const int y = threadIdx.y;

    out(x, y) = sin(in(x, y));
}


static __global__ void cos(cv::cuda::PtrStepSzf in, cv::cuda::PtrStepSzf out) {
    const int x = blockIdx.y;
    const int y = threadIdx.y;

    out(x, y) = cos(in(x, y));
}

__device__ int sign(float x)
{ 
	int t = x<0 ? -1 : 0;
	return x > 0 ? 1 : t;
}

static __global__ void wrap(cv::cuda::PtrStepSzf in) {
    const int x = blockIdx.y;
    const int y = threadIdx.y;

    while(abs(in(x, y)) > M_PI){
        in(x, y) -= sign(in(x, y)) * 2 * M_PI;
    }
}

static __constant__ float pre = 0.5 / M_PI;
static __constant__ float post = 2 * M_PI;

static __global__ void temporal_unwrap(cv::cuda::PtrStepSzf lf, cv::cuda::PtrStepSzf hf, 
                                 cv::cuda::PtrStepSzf out, float scale) {
    const int x = blockIdx.y;
    const int y = threadIdx.y;

    out(x, y) = (rint((lf(x, y)*scale - hf(x, y)) * pre) * post + hf(x, y));
}

static __global__ void round(cv::cuda::PtrStepSzf in, cv::cuda::PtrStepSzf out) {
    const int x = blockIdx.y;
    const int y = threadIdx.y;

    out(x, y) = rint(in(x, y) * pre) * post;
}

static __global__ void sin_cos(const cv::cuda::PtrStepSzf in,
                               cv::cuda::PtrStepSzf sin_out,
                               cv::cuda::PtrStepSzf cos_out) {
    const int x = blockIdx.y;
    const int y = threadIdx.y;

    sin_out(x, y) = sin(in(x, y));
    cos_out(x, y) = cos(in(x, y));
}

static __global__ void img_atan_inplace(cv::cuda::PtrStepSzf in0,
                                        const cv::cuda::PtrStepSzf in1,
                                        const cv::cuda::PtrStepSzf in2,
                                        const cv::cuda::PtrStepSzf in3,
                                        cv::cuda::PtrStepSzf out) {
    const int x = blockIdx.y;
    const int y = threadIdx.y;

    out(x, y) = atan2f(in3(x, y) - in1(x, y), in0(x, y) - in2(x, y));
}

static __global__ void vec_comp_elem_wise_mul2(
    cv::cuda::PtrStepSz<float2> in0,
    const cv::cuda::PtrStepSz<float2> dct_coeff, cv::cuda::PtrStepSzf out) {
    const int x = blockIdx.y;
    const int y = threadIdx.y;

    float2 in = in0(x, y);
    float2 coeff = dct_coeff(x, y);

    float res = in.x * coeff.x + in.y * coeff.y;

    out(x, y) = res;
}

static __global__ void vec_real_elem_wise_mul2(
    cv::cuda::PtrStepSzf in, const cv::cuda::PtrStepSz<float2> idct_coeff,
    cv::cuda::PtrStepSz<float2> out) {
    const int x = blockIdx.y;
    const int y = threadIdx.y;

    float in0 = in(x, y);
    float2 coeff0 = idct_coeff(x, y);
    float2 r0;

    r0.x = in0 * coeff0.x;
    r0.y = in0 * coeff0.y;

    out(x, y) = r0;
}

static __global__ void delta_phi_inplace(cv::cuda::PtrStepSzf in0,
                                         const cv::cuda::PtrStepSzf in1,
                                         const cv::cuda::PtrStepSzf cos_coeff,
                                         const cv::cuda::PtrStepSzf sin_coeff) {
    const int x = blockIdx.y;
    const int y = threadIdx.y;

    in0(x, y) = in0(x, y) * cos_coeff(x, y) - in1(x, y) * sin_coeff(x, y);
}

void cuda_invert_array(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y,
                 cv::cuda::GpuMat &z) {
    const dim3 block(1, x.cols / 2);
    const dim3 grid(1, x.rows);

    invert2D<<<grid, block>>>(x, y, z);
}

void cuda_sin(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y) {
    const dim3 block(1, x.cols);
    const dim3 grid(1, x.rows);

    sin<<<grid, block>>>(x, y);
}

void cuda_cos(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y) {
    const dim3 block(1, x.cols);
    const dim3 grid(1, x.rows);

    cos<<<grid, block>>>(x, y);
}

void cuda_round(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y) {
    const dim3 block(1, x.cols);
    const dim3 grid(1, x.rows);

    round<<<grid, block>>>(x, y);
}

void cuda_diff_atan_inplace(
    const std::vector<cv::cuda::GpuMat> &input,
    cv::cuda::GpuMat& out) {
    const dim3 block(1, input[0].cols);
    const dim3 grid(1, input[0].rows);

    img_atan_inplace<<<grid, block>>>(input[0], input[1], input[2],
                                      input[3], out);
}

void cuda_dft2dct_out_convert(const cv::cuda::GpuMat &d_input,
                              const cv::cuda::GpuMat &d_dct_coeff,
                              cv::cuda::GpuMat &d_out) {
    const dim3 block(1, d_input.cols);
    const dim3 grid(1, d_input.rows);

    vec_comp_elem_wise_mul2<<<grid, block>>>(d_input, d_dct_coeff,
                                             d_out.reshape(1));
}

void cuda_idft2idct_in_convert(const cv::cuda::GpuMat &d_input,
                               const cv::cuda::GpuMat &d_idct_coeff,
                               cv::cuda::GpuMat &d_out) {
    const dim3 block(1, d_input.cols);
    const dim3 grid(1, d_input.rows);

    vec_real_elem_wise_mul2<<<grid, block>>>(d_input, d_idct_coeff, d_out);
}

void cuda_delta_phi_mult_sub_inplace(cv::cuda::GpuMat &in1,
                                     const cv::cuda::GpuMat &in2,
                                     const cv::cuda::GpuMat &d_cos_f,
                                     const cv::cuda::GpuMat &d_sin_f) {
    const dim3 block(1, in1.cols);
    const dim3 grid(1, in1.rows);

    delta_phi_inplace<<<grid, block>>>(in1, in2, d_cos_f, d_sin_f);
}

void cuda_sin_cos(const cv::cuda::GpuMat &in, cv::cuda::GpuMat &sin_out,
                  cv::cuda::GpuMat &cos_out) {
    const dim3 block(1, in.cols);
    const dim3 grid(1, in.rows);

    sin_cos<<<grid, block>>>(in, sin_out, cos_out);
}

void cuda_temporal_unwrap(cv::cuda::GpuMat &x, cv::cuda::GpuMat &y, cv::cuda::GpuMat &z, float scale) {
    const dim3 block(1, x.cols);
    const dim3 grid(1, x.rows);

    temporal_unwrap<<<grid, block>>>(x, y, z, scale);
}

void cuda_wrap(cv::cuda::GpuMat &x) {
    const dim3 block(1, x.cols);
    const dim3 grid(1, x.rows);

    wrap<<<grid, block>>>(x);
}
