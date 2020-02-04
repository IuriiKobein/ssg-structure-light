#include <cufft.h>
#include <npp.h>
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
#include <vector>

#include <cxxopts.hpp>

#include "alg_utils.hpp"
#include "cl_alg.hpp"

namespace {
enum alg_type { PCG = 1, TEMPORAL_PHASE_UNWRAP = 2 };

struct alg_env {
    std::vector<cv::cuda::GpuMat> lf_img;
    std::vector<cv::cuda::GpuMat> hf_img;
};

void alg_3dr_temporal_pu_ref_upload(structure_light_alg& alg,
                                    cxxopts::ParseResult& input, alg_env& env) {
    env.lf_img =
        cuda_imgs_from_dir_load(input["lf_ref"].as<std::string>().c_str());
    env.hf_img =
        cuda_imgs_from_dir_load(input["hf_ref"].as<std::string>().c_str());

    alg.ref_phase_compute(env.lf_img, 0);
    alg.ref_phase_compute(env.hf_img, 1);
}

void alg_3dr_temporal_pu_src_upload(structure_light_alg& alg,
                                    cxxopts::ParseResult& input, alg_env& env) {
    env.lf_img =
        cuda_imgs_from_dir_load(input["lf_tar"].as<std::string>().c_str());
    env.hf_img =
        cuda_imgs_from_dir_load(input["hf_tar"].as<std::string>().c_str());

    alg.obj_phase_compute(env.lf_img, 0);
    alg.obj_phase_compute(env.hf_img, 1);
}

void alg_3dr_pcg_pu_ref_upload(structure_light_alg& alg,
                               cxxopts::ParseResult& input, alg_env& env) {
    env.hf_img =
        cuda_imgs_from_dir_load(input["hf_ref"].as<std::string>().c_str());

    alg.ref_phase_compute(env.hf_img, 0);
}

void alg_3dr_pcg_pu_src_upload(structure_light_alg& alg,
                               cxxopts::ParseResult& input, alg_env& env) {
    env.hf_img =
        cuda_imgs_from_dir_load(input["hf_tar"].as<std::string>().c_str());

    alg.obj_phase_compute(env.hf_img, 0);
}

void alg_3dr_imgs_upload(structure_light_alg& alg, cxxopts::ParseResult& input,
                         alg_env& env, int type) {
    if (type == TEMPORAL_PHASE_UNWRAP) {
        alg_3dr_temporal_pu_ref_upload(alg, input, env);
        alg_3dr_temporal_pu_src_upload(alg, input, env);
    } else if (type == PCG) {
        alg_3dr_pcg_pu_ref_upload(alg, input, env);
        alg_3dr_pcg_pu_src_upload(alg, input, env);
    }
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
            "lf_tar", "path to low freq target images",
            cxxopts::value<std::string>())("hf_ref",
                                           "path to high freq reference images",
                                           cxxopts::value<std::string>())(
            "hf_tar", "path to high freq target images",
            cxxopts::value<std::string>())(
            "c, counter", "number of repetitions", cxxopts::value<int>(c));

        auto result = opt.parse(argc, argv);

        alg_env env;
        structure_light_alg sla(cv::Size(h, w), t);
        alg_3dr_imgs_upload(sla, result, env, t);

        auto out = sla.compute_3d_reconstruction(t);

        int i = c;
        while (i--) {
            auto ts = std::chrono::high_resolution_clock::now();
            out = sla.compute_3d_reconstruction(t);
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
