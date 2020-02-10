#include <stdio.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <opencv2/core/mat.hpp>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/phase_unwrapping.hpp>

#include <string>
#include <type_traits>
#include <vector>

#include <cxxopts.hpp>

#include "alg_utils.hpp"
#include "sl_pcg_alg.hpp"
#include "sl_tpu_alg.hpp"

namespace {
enum alg_type { PCG = 1, TPU = 2 };

void alg_3dr_tpu_ref_upload(sl_alg& alg, cxxopts::ParseResult& input) {
    auto lf_ref = cuda_imgs_from_dir_load(input["lf_ref"].as<std::string>());
    auto hf_ref = cuda_imgs_from_dir_load(input["hf_ref"].as<std::string>());

    lf_ref.insert(lf_ref.end(), std::make_move_iterator(hf_ref.begin()),
                  std::make_move_iterator(hf_ref.end()));
    alg.ref_phase_compute(lf_ref);
}

void alg_3dr_tpu_obj_upload(sl_alg& alg, cxxopts::ParseResult& input) {
    auto lf_obj = cuda_imgs_from_dir_load(input["lf_obj"].as<std::string>());
    auto hf_obj = cuda_imgs_from_dir_load(input["hf_obj"].as<std::string>());

    lf_obj.insert(lf_obj.end(), std::make_move_iterator(hf_obj.begin()),
                  std::make_move_iterator(hf_obj.end()));
    alg.obj_phase_compute(lf_obj);
}

void alg_3dr_pcg_pu_ref_upload(sl_alg& alg, cxxopts::ParseResult& input) {
    auto hf_img = cuda_imgs_from_dir_load(input["hf_ref"].as<std::string>());

    alg.ref_phase_compute(hf_img);
}

void alg_3dr_pcg_pu_obj_upload(sl_alg& alg, cxxopts::ParseResult& input) {
    auto hf_obj = cuda_imgs_from_dir_load(input["hf_obj"].as<std::string>());

    alg.obj_phase_compute(hf_obj);
}

void alg_3dr_imgs_upload(sl_alg& alg, cxxopts::ParseResult& input, int type) {
    if (type == TPU) {
        alg_3dr_tpu_ref_upload(alg, input);
        alg_3dr_tpu_obj_upload(alg, input);
    } else if (type == PCG) {
        alg_3dr_pcg_pu_ref_upload(alg, input);
        alg_3dr_pcg_pu_obj_upload(alg, input);
    }
}

std::unique_ptr<sl_alg> sl_alg_create(int type, cv::Size size) {
    if (type == PCG) {
        return std::make_unique<sl_pcg_alg>(size);
    } else if (type == TPU) {
        return std::make_unique<sl_tpu_alg>(size);
    }

    return nullptr;
}
}  // namespace

// command to run ./build/3dr 1024 1024 test_images/lf/ref test_images/lf/phase
// test_images/hf/ref test_images/hf/phase 4
int main(int argc, char* argv[]) {
    int h, w, t, c;

    try {
        cxxopts::Options opt(
            argv[0], " - 3d reconstruction based on structured light techique");
        opt.add_options()("h,height", "images height", cxxopts::value<int>(h))(
            "w, width", "images width", cxxopts::value<int>(w))(
            "t, type", "type of phase unwrap algotirhm",
            cxxopts::value<int>(t))("lf_ref",
                                    "path to low freq reference images",
                                    cxxopts::value<std::string>())(
            "lf_obj", "path to low freq target images",
            cxxopts::value<std::string>())("hf_ref",
                                           "path to high freq reference images",
                                           cxxopts::value<std::string>())(
            "hf_obj", "path to high freq target images",
            cxxopts::value<std::string>())(
            "c, counter", "number of repetitions", cxxopts::value<int>(c));

        auto result = opt.parse(argc, argv);

        auto alg = sl_alg_create(t, {h, w});
        alg_3dr_imgs_upload(*alg, result, t);

        auto out = alg->compute_3d_reconstruction();

        int i = c;
        while (i--) {
            auto ts = std::chrono::high_resolution_clock::now();
            out = alg->compute_3d_reconstruction();
            auto te = std::chrono::high_resolution_clock::now();
            std::cout << std::chrono::duration_cast<std::chrono::microseconds>(
                             te - ts)
                             .count()
                      << std::endl;
        }

        img_show("cuda", out);
        cv::waitKey(0);

    } catch (const std::exception& e) {
        std::cout << e.what();
    }

    return 0;
}
