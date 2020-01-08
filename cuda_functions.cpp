#include "cuda_functions.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include "cuda_kernels.h"

namespace {

void cuda_dft2dct_convert2(const cv::cuda::GpuMat &dft_out,
                           ConstData &const_grid, VarMats &varMats,
                           cv::cuda::GpuMat &out) {
    const auto &cos_coeff = const_grid.cudaCosDCT;
    const auto &sin_coeff = const_grid.cudaSinDCT;
    auto &c_arr = varMats.c_arr;

    cv::cuda::split(dft_out, c_arr);

    cv::cuda::multiply(c_arr[0], cos_coeff, c_arr[0], 1, -1, varMats.s1);
    cv::cuda::multiply(c_arr[1], sin_coeff, c_arr[1], 1, -1, varMats.s2);
    varMats.s1.waitForCompletion();
    varMats.s2.waitForCompletion();

    cv::cuda::add(c_arr[0], c_arr[1], out);
}

void cuda_idft2idct_convert2(const cv::cuda::GpuMat &in, ConstData &const_grid,
                             VarMats &varMats, cv::cuda::GpuMat &out) {
    const auto &cos_coeff = const_grid.cudaCosDCT;
    const auto &sin_coeff = const_grid.cudaSinDCT;
    auto &c_arr = varMats.c_arr;

    cv::cuda::multiply(in, cos_coeff, c_arr[0], 1, -1, varMats.s1);
    cv::cuda::multiply(in, sin_coeff, c_arr[1], 1, -1, varMats.s2);
    varMats.s1.waitForCompletion();
    varMats.s2.waitForCompletion();

    cv::cuda::merge(c_arr, out);
}

cv::cuda::GpuMat &cuda_dct2(cv::cuda::GpuMat &img, ConstData &constGrids,
                            VarMats &varMats) {
    /* 0. extract preallocated contstant and temp vars */
    auto &fft_in = varMats.doubledMat;
    auto &fft_out = varMats.fftOut;
    auto &carr = varMats.c_arr;
    const auto &cos_coeff = constGrids.cudaCosDCT;
    const auto &sin_coeff = constGrids.cudaSinDCT;
    const auto h = constGrids.height;
    const auto w = constGrids.width;

    /* 1. to calcualte dct via fft make input signal
     * event related to left right corner */
    img.copyTo(fft_in(cv::Rect(0, 0, h, w)));
    cv::cuda::flip(img, fft_in(cv::Rect(0, w, h, w)), 0);
    cv::cuda::flip(img, fft_in(cv::Rect(h, 0, h, w)), 1);
    cv::cuda::flip(img, fft_in(cv::Rect(h, w, h, w)), -1);

    /* 2. apply real -> complex dft  */
    cv::cuda::dft(fft_in, fft_out, fft_in.size());

    /* 3. crop roi of dct */
    const auto &crop_fft_out = fft_out(cv::Rect(0, 0, h, w));

    //* 4. convert dft out to dct by twiddle factors*/
    carr[0].setTo(0);
    cuda_dft2dct_out_convert2(crop_fft_out, cos_coeff, sin_coeff, carr[0],
                              cv::cuda::StreamAccessor::getStream(varMats.s1),
                              cv::cuda::StreamAccessor::getStream(varMats.s2));

    return carr[0];
}

cv::cuda::GpuMat &idct(cv::cuda::GpuMat &img, ConstData &constGrids,
                       VarMats &varMats) {
    /* 0. extract preallocated contstant and temp vars */
    auto &ca = varMats.ca;
    auto &ifft_in = varMats.ifftIn;
    auto &c_arr = varMats.c_arr;
    auto &mat = varMats.Mat;
    const auto &cos_coeff = constGrids.cudaCosIDCT;
    const auto &sin_coeff = constGrids.cudaSinIDCT;

    //* 4. convert dft in to dct by twiddle factors*/
    cuda_idft2idct_in_convert2(img, cos_coeff, sin_coeff, ifft_in,
                              cv::cuda::StreamAccessor::getStream(varMats.s1),
                              cv::cuda::StreamAccessor::getStream(varMats.s2));

    cv::cuda::dft(ifft_in, ifft_in, ifft_in.size(),
                  cv::DFT_ROWS + cv::DFT_INVERSE + cv::DFT_SCALE);
    cv::cuda::split(ifft_in, c_arr);

    c_arr[0].convertTo(c_arr[0], c_arr[0].type(), 512);

    cv::cuda::flip(c_arr[0], mat, 1);
    invertArray(c_arr[0], mat, ca);

    return ca;
}

cv::cuda::GpuMat &cuda_idct2(cv::cuda::GpuMat &img, ConstData &constGrids,
                             VarMats &varMats) {
    varMats.x.release();

    cv::cuda::transpose(idct(img, constGrids, varMats), varMats.x);
    cv::cuda::transpose(idct(varMats.x, constGrids, varMats), varMats.x);

    return varMats.x;
}

void cudaLaplacian(cv::cuda::GpuMat &img, ConstData &constGrids,
                   VarMats &varMats, cv::cuda::GpuMat &out) {
    auto &ca = varMats.ca;
    const auto &l_grid = constGrids.cudaGridLaplacian;
    const auto h = constGrids.height;
    const auto w = constGrids.width;

    cv::cuda::multiply(cuda_dct2(img, constGrids, varMats), l_grid, ca);
    auto &idct_out = cuda_idct2(ca, constGrids, varMats);
    idct_out.convertTo(out, idct_out.type(), -4 * M_PI * M_PI / (h * w));
}

cv::cuda::GpuMat &cudaiLaplacian(cv::cuda::GpuMat &img, ConstData &constGrids,
                                 VarMats &varMats) {
    auto &ca = varMats.ca;
    const auto &l_grid = constGrids.cudaGridLaplacian;
    const auto h = constGrids.height;
    const auto w = constGrids.width;

    cv::cuda::divide(cuda_dct2(img, constGrids, varMats), l_grid, ca);

    auto &idct_out = cuda_idct2(ca, constGrids, varMats);
    idct_out.convertTo(idct_out, idct_out.type(), (h * w) / (-4 * M_PI * M_PI));

    return idct_out;
}

cv::cuda::GpuMat &deltaPhi(cv::cuda::GpuMat &img, ConstData &constGrids,
                           VarMats &varMats) {
    auto &img_sin = varMats.imgSin;
    auto &img_cos = varMats.imgCos;
    auto &a1 = varMats.a1;
    auto &a2 = varMats.a2;

    cudaSin(img, img_sin);
    cudaLaplacian(img_sin, constGrids, varMats, a1);

    cudaCos(img, img_cos);
    cudaLaplacian(img_cos, constGrids, varMats, a2);

    cuda_delta_phi_mult_sub_inplace(a1, a2, img_cos, img_sin, a1);

    return cudaiLaplacian(a1, constGrids, varMats);
}

cv::Scalar cudaMean(cv::cuda::GpuMat &img) {
    return cv::Scalar(cv::cuda::sum(img)[0] / img.cols / img.rows);
}
}  // namespace

void phaseUnwrap(cv::cuda::GpuMat &img, ConstData &constGrids,
                 VarMats &varMats) {
    auto &k1 = varMats.k1;
    auto &phi1 = varMats.phi1;
    auto &phi2 = varMats.phi2;
    auto &error = varMats.error;

    phi1 = deltaPhi(img, constGrids, varMats);
    cv::cuda::subtract(phi1, cudaMean(phi1), phi1);
    cv::cuda::add(phi1, cudaMean(img), phi1);

    cv::cuda::subtract(phi1, img, k1);
    k1.convertTo(k1, k1.type(), 0.5 / M_PI);

    cudaRound(k1, k1);
    k1.convertTo(k1, k1.type(), 2 * M_PI);

    cv::cuda::add(img, k1, phi2);

    for (auto i = 0; i < 3; i++) {
        cv::cuda::subtract(phi2, phi1, error);
        cv::cuda::subtract(phi1, cudaMean(phi1), phi1);
        cv::cuda::add(phi1, deltaPhi(error, constGrids, varMats), phi1);
        cv::cuda::add(phi1, cudaMean(phi2), phi1);

        cv::cuda::subtract(phi1, img, k1);
        k1.convertTo(k1, k1.type(), 0.5 / M_PI);
        cudaRound(k1, k1);
        k1.convertTo(k1, k1.type(), 2 * M_PI);

        cv::cuda::add(img, k1, phi2);
    }

    img = varMats.phi2;
    /*//SAVING IMG
    std::string s1 = "Reconstruction";
    std::string s2 = ".bmp";
    cv::Mat cudaResult;
    varMats.phi2.download(cudaResult);
    double min, max;
    cv::minMaxLoc(cudaResult, &min, &max);
    cudaResult += 5;
    cv::normalize(cudaResult, cudaResult, 255, 0, cv::NORM_MINMAX);
    std::string s = std::to_string(1);
    cv::imwrite( s1+s+s2, cudaResult);
    */
}
