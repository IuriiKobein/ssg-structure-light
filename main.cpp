#include <cufft.h>
#include <npp.h>
#include <stdio.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/phase_unwrapping.hpp>

#include <string>
#include <vector>
#include "cuda_functions.h"
#include "cuda_kernels.h"

namespace {

const int IMG_WIDTH = 512;
const int IMG_HEIGHT = 512;

cv::Mat ReadMatFromTxt(std::string filename, int rows, int cols) {
    float m;
    cv::Mat out = cv::Mat::zeros(rows, cols, CV_32F);  // Matrix to store values

    std::ifstream fileStream(filename);
    int cnt = 0;  // index starts from 0
    while (fileStream >> m) {
        int temprow = cnt / cols;
        int tempcol = cnt % cols;
        out.at<float>(temprow, tempcol) = m;
        cnt++;
    }
    return out;
}

cv::Mat dct2(cv::Mat &img) {
    int height = img.rows;
    int width = img.cols;

    auto gridCos = cv::Mat(height, width, CV_32FC1);
    auto gridSin = cv::Mat(height, width, CV_32FC1);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (i > 0 && j > 0) {
                gridCos.at<float>(i, j) = (2 / sqrt(height * width) / 4) *
                                          cos(((j)*height + (i)*width) *
                                              (M_PI / (2 * width * height)));
                gridSin.at<float>(i, j) = (2 / sqrt(height * width) / 4) *
                                          sin(((j)*height + (i)*width) *
                                              (M_PI / (2 * width * height)));
            } else if (i == 0 && j > 0) {
                gridCos.at<float>(i, j) = (2 / (sqrt(2) * width) / 4) *
                                          cos(((j)*height + (i)*width) *
                                              (M_PI / (2 * width * height)));
                gridSin.at<float>(i, j) = (2 / (sqrt(2) * width) / 4) *
                                          sin(((j)*height + (i)*width) *
                                              (M_PI / (2 * width * height)));
            } else if (i > 0 && j == 0) {
                gridCos.at<float>(i, j) = (2 / (sqrt(2) * height) / 4) *
                                          cos(((j)*height + (i)*width) *
                                              (M_PI / (2 * width * height)));
                gridSin.at<float>(i, j) = (2 / (sqrt(2) * height) / 4) *
                                          sin(((j)*height + (i)*width) *
                                              (M_PI / (2 * width * height)));

            } else if (i == 0 && j == 0) {
                gridCos.at<float>(i, j) = (1 / sqrt(height * width) / 4) *
                                          cos(((j)*height + (i)*width) *
                                              (M_PI / (2 * width * height)));
                gridSin.at<float>(i, j) = (1 / sqrt(height * width) / 4) *
                                          sin(((j)*height + (i)*width) *
                                              (M_PI / (2 * width * height)));
            }
        }
    }

    auto y = cv::Mat(2 * height, 2 * width, CV_32FC1);
    img.copyTo(y(cv::Rect(0, 0, height, width)));
    cv::Mat tempMat;
    cv::flip(img, tempMat, 0);
    tempMat.copyTo(y(cv::Rect(0, width, height, width)));
    cv::flip(img, tempMat, 1);
    tempMat.copyTo(y(cv::Rect(height, 0, height, width)));
    cv::flip(img, tempMat, -1);
    tempMat.copyTo(y(cv::Rect(height, width, height, width)));

    cv::Mat fftOutput;
    cv::dft(y, fftOutput, cv::DFT_COMPLEX_OUTPUT);

    cv::Mat output;
    fftOutput(cv::Rect(0, 0, height, width)).copyTo(output);

    std::vector<cv::Mat> complexArray(2);
    cv::split(output, complexArray);

    cv::multiply(complexArray[0], gridCos, gridCos);
    cv::multiply(complexArray[1], gridSin, gridSin);

    cv::add(gridCos, gridSin, gridSin);

    std::cout << gridSin(cv::Rect(0, 0, 5, 5)) << std::endl;
    return gridSin;
}

cv::Mat idct(cv::Mat imgRow, bool flag = false) {
    int width = imgRow.cols;

    auto gridCos = cv::Mat(1, width, CV_32FC1);
    auto gridSin = cv::Mat(1, width, CV_32FC1);

    for (int i = 0; i < width; i++) {
        if (i > 0) {
            gridCos.at<float>(0, i) =
                sqrt(2 * width) * cos(M_PI * i / (2 * width));
            gridSin.at<float>(0, i) =
                sqrt(2 * width) * sin(M_PI * i / (2 * width));
        } else if (i == 0) {
            gridCos.at<float>(0, i) = sqrt(width) * cos(M_PI * i / (2 * width));
            gridSin.at<float>(0, i) = sqrt(width) * sin(M_PI * i / (2 * width));
        }
    }

    cv::multiply(imgRow, gridCos, gridCos);
    cv::multiply(imgRow, gridSin, gridSin);

    std::vector<cv::Mat> channels;
    cv::Mat merged;
    channels.push_back(gridCos);
    channels.push_back(gridSin);
    cv::merge(channels, merged);

    cv::dft(merged, merged, cv::DFT_INVERSE + cv::DFT_SCALE);

    std::vector<cv::Mat> complexArray(2);
    cv::split(merged, complexArray);

    cv::Mat real = complexArray[0];

    auto y = cv::Mat(1, width, CV_32FC1);
    cv::Mat tempMat;
    cv::flip(real, tempMat, 1);
    for (int i = 0; i < width / 2; i++) {
        y.at<float>(0, 2 * i) = real.at<float>(0, i);
        y.at<float>(0, 2 * i + 1) = tempMat.at<float>(0, i);
    }
    if (flag) std::cout << gridCos(cv::Rect(120, 0, 20, 1)) << " ";
    return y;
}

cv::Mat idct2(cv::Mat img) {
    int height = img.rows;
    int width = img.cols;

    cv::Mat b = cv::Mat(height, width, CV_32FC1);

    for (auto i = 0; i < height; i++) {
        if (i != 300)
            idct(img.row(i)).copyTo(b.row(i));
        else
            idct(img.row(i)).copyTo(b.row(i));
    }
    std::cout << std::endl;
    cv::transpose(b, b);
    for (auto i = 0; i < width; i++) {
        if (i != 300)
            idct(b.row(i)).copyTo(b.row(i));
        else
            idct(b.row(i)).copyTo(b.row(i));
    }

    cv::transpose(b, b);
    return b;
}

cv::Mat Laplacian(cv::Mat &img) {
    int height = img.rows;
    int width = img.cols;

    cv::Mat grid = cv::Mat(height, width, CV_32FC1);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            grid.at<float>(i, j) = (i + 1) * (i + 1) + (j + 1) * (j + 1);
        }
    }
    auto ca = dct2(img);

    cv::multiply(ca, grid, ca);
    ca = idct2(ca);
    ca *= -4 * M_PI * M_PI / (width * height);

    return ca;
}

cv::Mat iLaplacian(cv::Mat &img) {
    int height = img.rows;
    int width = img.cols;

    cv::Mat grid = cv::Mat(height, width, CV_32FC1);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            grid.at<float>(i, j) = (i + 1) * (i + 1) + (j + 1) * (j + 1);
        }
    }
    auto ca = dct2(img);
    cv::divide(ca, grid, ca);

    ca = idct2(ca);
    ca *= (width * height) / (-4 * M_PI * M_PI);

    return ca;
}

void CreateCudaGrids(cv::Size size, ConstData &constGrids) {
    int height = size.height;
    int width = size.width;

    auto grid_dct_twiddle = cv::Mat(height, width, CV_32FC2);
    auto grid_idct_twiddle = cv::Mat(height, width, CV_32FC2);
    auto gridLaplacian = cv::Mat(height, width, CV_32FC1);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (i > 0 && j > 0) {
                grid_dct_twiddle.at<float2>(i, j).x = (2 / sqrt(height * width) / 4) *
                                             cos(((j)*height + (i)*width) *
                                                 (M_PI / (2 * width * height)));
                grid_dct_twiddle.at<float2>(i, j).y = (2 / sqrt(height * width) / 4) *
                                             sin(((j)*height + (i)*width) *
                                                 (M_PI / (2 * width * height)));
            } else if (i == 0 && j > 0) {
                grid_dct_twiddle.at<float2>(i, j).x = (2 / (sqrt(2) * width) / 4) *
                                             cos(((j)*height + (i)*width) *
                                                 (M_PI / (2 * width * height)));
                grid_dct_twiddle.at<float2>(i, j).y = (2 / (sqrt(2) * width) / 4) *
                                             sin(((j)*height + (i)*width) *
                                                 (M_PI / (2 * width * height)));
            } else if (i > 0 && j == 0) {
                grid_dct_twiddle.at<float2>(i, j).x = (2 / (sqrt(2) * height) / 4) *
                                             cos(((j)*height + (i)*width) *
                                                 (M_PI / (2 * width * height)));
                grid_dct_twiddle.at<float2>(i, j).y = (2 / (sqrt(2) * height) / 4) *
                                             sin(((j)*height + (i)*width) *
                                                 (M_PI / (2 * width * height)));
            } else if (i == 0 && j == 0) {
                grid_dct_twiddle.at<float2>(i, j).x = (1 / sqrt(height * width) / 4) *
                                             cos(((j)*height + (i)*width) *
                                                 (M_PI / (2 * width * height)));
                grid_dct_twiddle.at<float2>(i, j).y = (1 / sqrt(height * width) / 4) *
                                             sin(((j)*height + (i)*width) *
                                                 (M_PI / (2 * width * height)));
            }
            if (j > 0) {
                grid_idct_twiddle.at<float2>(i, j).x =
                    sqrt(2 * width) * cos(M_PI * j / (2 * width));
                grid_idct_twiddle.at<float2>(i, j).y =
                    sqrt(2 * width) * sin(M_PI * j / (2 * width));
            } else if (j == 0) {
                grid_idct_twiddle.at<float2>(i, j).x =
                    sqrt(width) * cos(M_PI * j / (2 * width));
                grid_idct_twiddle.at<float2>(i, j).y =
                    sqrt(width) * sin(M_PI * j / (2 * width));
            }
            gridLaplacian.at<float>(i, j) =
                (i + 1) * (i + 1) + (j + 1) * (j + 1);
        }
    }
    constGrids.height = height;
    constGrids.width = width;
    constGrids.dct_twiddle.upload(grid_dct_twiddle);
    constGrids.idct_twiddle.upload(grid_idct_twiddle);
    constGrids.cudaGridLaplacian.upload(gridLaplacian);
}

void InitVars(VarMats &varMats, int height, int width) {
    varMats.doubledMat = cv::cuda::GpuMat(2 * height, 2 * width, CV_32FC1);
    varMats.Mat = cv::cuda::GpuMat(height, width, CV_32FC1);
    varMats.fftOut = cv::cuda::GpuMat(2 * height, width + 1, CV_32FC2);
    varMats.ifftIn = cv::cuda::GpuMat(height, width, CV_32FC2);
    varMats.imgSin = cv::cuda::GpuMat(height, width, CV_32FC1);
    varMats.imgCos = cv::cuda::GpuMat(height, width, CV_32FC1);
    varMats.ca = cv::cuda::GpuMat(height, width, CV_32FC1);
    varMats.a1 = cv::cuda::GpuMat(height, width, CV_32FC1);
    varMats.a2 = cv::cuda::GpuMat(height, width, CV_32FC1);
    varMats.k1 = cv::cuda::GpuMat(height, width, CV_32FC1);
    varMats.phi1 = cv::cuda::GpuMat(height, width, CV_32FC1);
    varMats.phi2 = cv::cuda::GpuMat(height, width, CV_32FC1);
    varMats.error = cv::cuda::GpuMat(height, width, CV_32FC1);
    varMats.x = cv::cuda::GpuMat(height, width, CV_32FC1);
    for (auto i = 0; i < 2; i++) {
        varMats.c_arr.push_back(varMats.Mat.clone());
    }
}

std::vector<cv::Mat> host_imgs_load(const std::vector<std::string> &path_list) {
    std::vector<cv::Mat> imgs;
    imgs.reserve(path_list.size());

    std::transform(
        std::begin(path_list), std::end(path_list), std::back_inserter(imgs),
        [](const auto &p) { return cv::imread(p, cv::IMREAD_GRAYSCALE); });

    return imgs;
}

std::vector<cv::cuda::GpuMat> cuda_imgs_alloc(std::size_t num, cv::Size size,
                                              int type) {
    std::vector<cv::cuda::GpuMat> d_imgs;
    d_imgs.reserve(num);

    for (auto i = 0ul; i < num; ++i) {
        d_imgs.emplace_back(size, type);
    }

    return d_imgs;
}

std::vector<cv::cuda::GpuMat> cuda_imgs_load(
    const std::vector<std::string> &path_list) {
    auto h_imgs = host_imgs_load(path_list);
    auto d_imgs = cuda_imgs_alloc(h_imgs.size(), h_imgs[0].size(), CV_8U);

    for (auto i = 0ul; i < h_imgs.size(); ++i) {
        d_imgs[i].upload(h_imgs[i]);
    }

    return d_imgs;
}

std::vector<cv::cuda::GpuMat> cuda_imgs_from_dir_load(const char *dir_path) {
    using namespace std::experimental;
    std::vector<std::string> path_list;

    for (const auto &entry : filesystem::directory_iterator(dir_path)) {
        path_list.emplace_back(entry.path());
    }

    std::sort(std::begin(path_list), std::end(path_list));

    for (const auto &p : path_list) {
        std::cout << p << '\n';
    }

    return cuda_imgs_load(path_list);
}

void img_show(const std::string title, cv::Mat &h_img) {
    cv::normalize(h_img, h_img, 0, 1, cv::NORM_MINMAX, CV_32F);
    cv::namedWindow(title, cv::WINDOW_NORMAL);
    cv::imshow(title, h_img);
}
void cuda_img_show(const std::string title, cv::cuda::GpuMat &d_img) {
    cv::Mat h_img(d_img.size(), d_img.type());

    d_img.download(h_img);
    img_show(title, h_img);
}
}  // namespace

int main(int argc, char *argv[]) {
    ConstData constGrids;
    VarMats varMats;

    CreateCudaGrids(cv::Size(512, 512), constGrids);
    InitVars(varMats, constGrids.height, constGrids.width);

    /*==============================================================================*/
    int i = 4;
    while (i--) {
        auto srcs_u8 = cuda_imgs_from_dir_load(argv[1]);
        auto refs_u8 = cuda_imgs_from_dir_load(argv[2]);

        auto refs_f32 =
            cuda_imgs_alloc(refs_u8.size(), refs_u8[0].size(), CV_32F);
        auto srcs_f32 =
            cuda_imgs_alloc(srcs_u8.size(), srcs_u8[0].size(), CV_32F);

        for (auto i = 0ul; i < refs_u8.size(); ++i) {
            refs_u8[i].convertTo(refs_f32[i], refs_f32[i].type(), 1.0f / 255);
            srcs_u8[i].convertTo(srcs_f32[i], srcs_f32[i].type(), 1.0f / 255);
        }

        auto filt =
            cv::cuda::createGaussianFilter(CV_32F, CV_32F, cv::Size(5, 5), 0);
        std::for_each(std::begin(refs_f32), std::end(refs_f32),
                      [&filt](auto &img) { filt->apply(img, img); });
        std::for_each(std::begin(srcs_f32), std::end(srcs_f32),
                      [&filt](auto &img) { filt->apply(img, img); });

        auto &ref_phase = cuda_diff_atan_inplace(refs_f32);
        auto &src_phase = cuda_diff_atan_inplace(srcs_f32);

        cv::cuda::subtract(src_phase, ref_phase, src_phase);

        auto ts = std::chrono::high_resolution_clock::now();
        auto te = std::chrono::high_resolution_clock::now();
        if (argc == 4 && *argv[3] == 'c') {
            cv::Mat h_in(cv::Size(512, 512), CV_32FC1),
                h_out(cv::Size(512, 512), CV_32FC1);

            src_phase.download(h_in);
            cv::phase_unwrapping::HistogramPhaseUnwrapping::Params params;
            params.width = 512;
            params.height = 512;
            auto pu =
                cv::phase_unwrapping::HistogramPhaseUnwrapping::create(params);

            ts = std::chrono::high_resolution_clock::now();

            pu->unwrapPhaseMap(h_in, h_out);

            te = std::chrono::high_resolution_clock::now();
            img_show("unwrapped", h_out);
            std::cout << "cpu phase unwrap:";
        } else {
            ts = std::chrono::high_resolution_clock::now();
            phaseUnwrap(src_phase, constGrids, varMats);
            te = std::chrono::high_resolution_clock::now();
            cuda_img_show("unwrapped", src_phase);
            std::cout << "cuda phase unwrap:";
        }
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(te -
                                                                           ts)
                         .count()
                  << std::endl;
    }
    cv::waitKey(0);

    return 0;
}
