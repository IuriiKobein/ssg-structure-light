#include "cuda_phase_unwrap.hpp"
#include "cuda_kernels.h"

#include "alg_utils.hpp"

#include <cufft.h>
#include <memory>
#include <opencv2/core/base.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/cudaarithm.hpp>

#include <iostream>

namespace {

struct alg_const_mats {
    cv::Size size;
    cv::cuda::GpuMat dct_f;
    cv::cuda::GpuMat idct_f;
    cv::cuda::GpuMat laplacian;

    alg_const_mats(cv::Size s, cv::Mat dct, cv::Mat idct, cv::Mat lapl)
        : size(s) {
        dct_f.upload(dct);
        idct_f.upload(idct);
        laplacian.upload(lapl);
    }
};

struct alg_tmp_mats {
    cv::cuda::GpuMat doubled_mat;
    cv::cuda::GpuMat mat;
    std::vector<cv::cuda::GpuMat> c_arr;
    cv::cuda::GpuMat img_sin;
    cv::cuda::GpuMat img_cos;
    cv::cuda::GpuMat fft_out;
    cv::cuda::GpuMat ifft_in;
    cv::cuda::GpuMat ca;
    cv::cuda::GpuMat a1;
    cv::cuda::GpuMat a2;
    cv::cuda::GpuMat k1;
    cv::cuda::GpuMat k2;
    cv::cuda::GpuMat phi1;
    cv::cuda::GpuMat phi2;
    cv::cuda::GpuMat error;
    cv::cuda::GpuMat x;
};

struct alg_env {
    cv::cuda::DFT *dft, *idft;
    const alg_const_mats *c_mats;
    alg_tmp_mats *t_mats;
};

static alg_env g_alg_env;

alg_const_mats alg_make_const_mat(cv::Size size) {
    int h = size.height;
    int w = size.width;

    auto dct_f = cv::Mat(size, CV_32FC2);
    auto idct_f = cv::Mat(size, CV_32FC2);
    auto laplacian = cv::Mat(size, CV_32FC1);

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            auto cos_f = cos((j * h + i * w) * (M_PI / (2 * w * h)));
            auto sin_f = sin((j * h + i * w) * (M_PI / (2 * w * h)));

            if (i > 0 && j > 0) {
                dct_f.at<float2>(i, j).x = (2 / sqrt(h * w) / 4) * cos_f;
                dct_f.at<float2>(i, j).y = (2 / sqrt(h * w) / 4) * sin_f;
            } else if (i == 0 && j > 0) {
                dct_f.at<float2>(i, j).x = (2 / (sqrt(2) * w) / 4) * cos_f;
                dct_f.at<float2>(i, j).y = (2 / (sqrt(2) * w) / 4) * sin_f;
            } else if (i > 0 && j == 0) {
                dct_f.at<float2>(i, j).x = (2 / (sqrt(2) * h) / 4) * cos_f;
                dct_f.at<float2>(i, j).y = (2 / (sqrt(2) * h) / 4) * sin_f;
            } else if (i == 0 && j == 0) {
                dct_f.at<float2>(i, j).x = (1 / sqrt(h * w) / 4) * cos_f;
                dct_f.at<float2>(i, j).y = (1 / sqrt(h * w) / 4) * sin_f;
            }
            if (j > 0) {
                idct_f.at<float2>(i, j).x =
                    sqrt(2 * w) * cos(M_PI * j / (2 * w));
                idct_f.at<float2>(i, j).y =
                    sqrt(2 * w) * sin(M_PI * j / (2 * w));
            } else if (j == 0) {
                idct_f.at<float2>(i, j).x = sqrt(w) * cos(M_PI * j / (2 * w));
                idct_f.at<float2>(i, j).y = sqrt(w) * sin(M_PI * j / (2 * w));
            }
            laplacian.at<float>(i, j) = (i + 1) * (i + 1) + (j + 1) * (j + 1);
        }
    }

    return {size, dct_f, idct_f, laplacian};
}

alg_tmp_mats alg_make_tmp_mat(cv::Size size) {
    alg_tmp_mats mats;

    mats.doubled_mat =
        cv::cuda::GpuMat(2 * size.height, 2 * size.width, CV_32FC1);
    mats.mat = cv::cuda::GpuMat(size, CV_32FC1);
    mats.fft_out = cv::cuda::GpuMat(2 * size.height, size.width + 1, CV_32FC2);
    mats.ifft_in = cv::cuda::GpuMat(size, CV_32FC2);
    mats.img_sin = cv::cuda::GpuMat(size, CV_32FC1);
    mats.img_cos = cv::cuda::GpuMat(size, CV_32FC1);
    mats.ca = cv::cuda::GpuMat(size, CV_32FC1);
    mats.a1 = cv::cuda::GpuMat(size, CV_32FC1);
    mats.a2 = cv::cuda::GpuMat(size, CV_32FC1);
    mats.k1 = cv::cuda::GpuMat(size, CV_32FC1);
    mats.phi1 = cv::cuda::GpuMat(size, CV_32FC1);
    mats.phi2 = cv::cuda::GpuMat(size, CV_32FC1);
    mats.error = cv::cuda::GpuMat(size, CV_32FC1);
    mats.x = cv::cuda::GpuMat(size, CV_32FC1);
    for (auto i = 0; i < 2; i++) {
        mats.c_arr.push_back(mats.mat.clone());
    }

    return mats;
}

void cuda_dct2(cv::cuda::GpuMat &img, cv::cuda::GpuMat &out) {
    /* 0. extract preallocated contstant and temp vars */
    auto &fft_in = g_alg_env.t_mats->doubled_mat;
    auto &fft_out = g_alg_env.t_mats->fft_out;
    const auto &dct_coeff = g_alg_env.c_mats->dct_f;
    const auto h = g_alg_env.c_mats->size.height;
    const auto w = g_alg_env.c_mats->size.width;

    /* 1. to calcualte dct via fft make input signal
     * event related to left right corner */
    img.copyTo(fft_in(cv::Rect(0, 0, h, w)));
    cv::cuda::flip(img, fft_in(cv::Rect(0, w, h, w)), 0);
    cv::cuda::flip(img, fft_in(cv::Rect(h, 0, h, w)), 1);
    cv::cuda::flip(img, fft_in(cv::Rect(h, w, h, w)), -1);

    /* 2. apply real -> complex dft  */
    g_alg_env.dft->compute(fft_in, fft_out);

    /* 3. crop roi of dct */
    const auto &crop_fft_out = fft_out(cv::Rect(0, 0, h, w));

    //* 4. convert dft out to dct by twiddle factors*/
    cuda_dft2dct_out_convert(crop_fft_out, dct_coeff, out);
}

cv::cuda::GpuMat &idct(cv::cuda::GpuMat &img) {
    /* 0. extract preallocated contstant and temp vars */
    auto &ca = g_alg_env.t_mats->ca;
    auto &ifft_in = g_alg_env.t_mats->ifft_in;
    auto &c_arr = g_alg_env.t_mats->c_arr;
    auto &mat = g_alg_env.t_mats->mat;
    const auto &idct_coeff = g_alg_env.c_mats->idct_f;
    const auto h = g_alg_env.c_mats->size.height;

    //* 1. convert dft in to dct by twiddle factors*/
    cuda_idft2idct_in_convert(img, idct_coeff, ifft_in);

    g_alg_env.idft->compute(ifft_in, ifft_in);
    cv::cuda::split(ifft_in, c_arr);

    c_arr[0].convertTo(c_arr[0], c_arr[0].type(), h);

    cv::cuda::flip(c_arr[0], mat, 1);
    cuda_invert_array(c_arr[0], mat, ca);

    return ca;
}

void cuda_idct2(cv::cuda::GpuMat &img, cv::cuda::GpuMat &out) {
    out.release();

    cv::cuda::transpose(idct(img), out);
    cv::cuda::transpose(idct(out), out);
}

void cuda_laplacian(cv::cuda::GpuMat &img, cv::cuda::GpuMat &out) {
    auto &ca = g_alg_env.t_mats->ca;
    const auto &l_grid = g_alg_env.c_mats->laplacian;

    cuda_dct2(img, ca);
    cv::cuda::multiply(ca, l_grid, ca);
    cuda_idct2(ca, out);
}

void cudaiLaplacian(cv::cuda::GpuMat &img, cv::cuda::GpuMat &out) {
    auto &ca = g_alg_env.t_mats->ca;
    const auto &l_grid = g_alg_env.c_mats->laplacian;

    cuda_dct2(img, ca);
    cv::cuda::divide(ca, l_grid, ca);
    cuda_idct2(ca, out);
}

void delta_phi(cv::cuda::GpuMat &img, cv::cuda::GpuMat &out) {
    auto &img_sin = g_alg_env.t_mats->img_sin;
    auto &img_cos = g_alg_env.t_mats->img_cos;
    auto &a1 = g_alg_env.t_mats->a1;
    auto &a2 = g_alg_env.t_mats->a2;

    cuda_sin(img, img_sin);
    cuda_laplacian(img_sin, a1);

    cuda_cos(img, img_cos);
    cuda_laplacian(img_cos, a2);

    cuda_delta_phi_mult_sub_inplace(a1, a2, img_cos, img_sin);

    cudaiLaplacian(a1, out);
}

void cuda_phase_unwrap(cv::cuda::GpuMat &img) {
    auto &k1 = g_alg_env.t_mats->k1;
    auto &phi1 = g_alg_env.t_mats->phi1;
    auto &phi2 = g_alg_env.t_mats->phi2;
    auto &error = g_alg_env.t_mats->error;

    delta_phi(img, phi1);
    cv::cuda::subtract(phi1, img, k1);
    cuda_round(k1, k1);
    cv::cuda::add(img, k1, phi2);

    for (auto i = 0; i < 1; i++) {
        cv::cuda::subtract(phi2, phi1, error);
        delta_phi(error, error);
        cv::cuda::add(phi1, error, phi1);
        cv::cuda::subtract(phi1, img, k1);
        cuda_round(k1, k1);
        cv::cuda::add(img, k1, phi2);
    }

    img = phi2;
}
void cuda_temp_unwrap(cv::cuda::GpuMat &phase1, cv::cuda::GpuMat &phase2,
                      float scale) {
    // add filering?

    auto &phi = g_alg_env.t_mats->phi2;

    cuda_wrap(phase1);
    cuda_temporal_unwrap(phase1, phase2, phi, scale);

    phase2 = phi;
}
}  // namespace

class cuda_phase_unwrap_alg::cu_pu_impl {
   public:
    cu_pu_impl(cv::Size size)
        : _const_mats(alg_make_const_mat(size)),
          _tmp_mats(alg_make_tmp_mat(size)),
          _dft(cv::cuda::createDFT(cv::Size(size.height * 2, size.width * 2),
                                   0)),
          _idft(cv::cuda::createDFT(size, cv::DFT_COMPLEX_INPUT | cv::DFT_ROWS |
                                              cv::DFT_INVERSE | cv::DFT_SCALE))

    {
        g_alg_env.c_mats = &_const_mats;
        g_alg_env.t_mats = &_tmp_mats;
        g_alg_env.dft = _dft.get();
        g_alg_env.idft = _idft.get();
    }

    cv::Mat gradient_unwrap(cv::cuda::GpuMat &in) {
        cv::Mat out;

        cuda_phase_unwrap(in);
        in.download(out);

        return out;
    }

    cv::Mat temporal_unwrap(cv::cuda::GpuMat &in1, cv::cuda::GpuMat &in2,
                            float scale) {
        cv::Mat out;

        cuda_temp_unwrap(in1, in2, scale);
        in2.download(out);

        return out;
    }

   private:
    const alg_const_mats _const_mats;
    alg_tmp_mats _tmp_mats;
    cv::Ptr<cv::cuda::DFT> _dft;
    cv::Ptr<cv::cuda::DFT> _idft;
};

cuda_phase_unwrap_alg::cuda_phase_unwrap_alg(cv::Size size)
    : _pimpl(std::make_unique<cu_pu_impl>(size)) {}

cuda_phase_unwrap_alg::~cuda_phase_unwrap_alg() = default;

cv::Mat cuda_phase_unwrap_alg::gradient_unwrap(cv::cuda::GpuMat &in) {
    return _pimpl->gradient_unwrap(in);
}

cv::Mat cuda_phase_unwrap_alg::temporal_unwrap(cv::cuda::GpuMat &in1,
                                               cv::cuda::GpuMat &in2,
                                               float scale) {
    return _pimpl->temporal_unwrap(in1, in2, scale);
}
