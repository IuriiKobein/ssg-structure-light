#include <algorithm>
#include <chrono>
#include <exception>
#include <iostream>
#include <memory>
#include <opencv2/core/mat.hpp>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <string>
#include <type_traits>
#include <vector>

#include <cxxopts.hpp>

#include "alg_utils.hpp"
#include "sl_alg_factory.hpp"

namespace {

void alg_3dr_tpu_ref_upload(sl_alg& alg, cxxopts::ParseResult& input) {
    auto lf_ref = imgs_from_dir_load(input["lf_ref"].as<std::string>());
    auto hf_ref = imgs_from_dir_load(input["hf_ref"].as<std::string>());

    alg.ref_phase_compute(lf_ref, hf_ref);
}

cv::Mat alg_3dr_tpu_obj_depth_compute(sl_alg& alg,
                                      cxxopts::ParseResult& input) {
    auto lf_obj = imgs_from_dir_load(input["lf_obj"].as<std::string>());
    auto hf_obj = imgs_from_dir_load(input["hf_obj"].as<std::string>());

    return alg.depth_compute(lf_obj, hf_obj);
}

void alg_3dr_pcg_ref_upload(sl_alg& alg, cxxopts::ParseResult& input) {
    auto hf_img = imgs_from_dir_load(input["hf_ref"].as<std::string>());

    alg.ref_phase_compute(hf_img);
}

cv::Mat alg_3dr_pcg_obj_depth_compute(sl_alg& alg,
                                      cxxopts::ParseResult& input) {
    auto ts = std::chrono::high_resolution_clock::now();
    auto hf_obj = imgs_from_dir_load(input["hf_obj"].as<std::string>());
    auto te = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(te - ts)
                     .count()
              << std::endl;

    return alg.depth_compute(hf_obj);
}

void alg_3dr_imgs_ref_compute(sl_alg& alg, cxxopts::ParseResult& input,
                              int type) {
    if (type == TPU) {
        alg_3dr_tpu_ref_upload(alg, input);
    } else if (type == PCG) {
        alg_3dr_pcg_ref_upload(alg, input);
    }
}

cv::Mat alg_3dr_obj_depth_compute(sl_alg& alg, cxxopts::ParseResult& input,
                                  int type) {
    if (type == TPU) {
        return alg_3dr_tpu_obj_depth_compute(alg, input);
    } else if (type == PCG) {
        return alg_3dr_pcg_obj_depth_compute(alg, input);
    }

    return cv::Mat();
}

int sla_run_once(int alg_type, sl_alg* alg, cxxopts::ParseResult& parse_res,
                 int counter) {
    alg_3dr_imgs_ref_compute(*alg, parse_res, alg_type);

    auto out = alg_3dr_obj_depth_compute(*alg, parse_res, alg_type);

    int i = counter;
    while (i--) {
        auto ts = std::chrono::high_resolution_clock::now();
        out = alg_3dr_obj_depth_compute(*alg, parse_res, alg_type);
        auto te = std::chrono::high_resolution_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(te -
                                                                           ts)
                         .count()
                  << std::endl;
    }

    img_show("cuda", out);
    cv::waitKey(0);

    return 0;
}
}  // namespace

// command to run ./build/3dr 1024 1024 test_images/lf/ref test_images/lf/phase
// test_images/hf/ref test_images/hf/phase 4
int main(int argc, char* argv[]) {
    int h, w, c;
    std::string m;

    try {
        cxxopts::Options opt(
            argv[0], " - 3d reconstruction based on structured light techique");
        opt.add_options()("h,height", "images height", cxxopts::value<int>(h))(
            "w, width", "images width", cxxopts::value<int>(w))(
            "method", "method for phase unwrap", cxxopts::value<std::string>())(
            "rpc", "run 3dr in interactive mode as rpc server")(
            "lf_ref", "path to low freq reference images",
            cxxopts::value<std::string>())("lf_obj",
                                           "path to low freq target images",
                                           cxxopts::value<std::string>())(
            "hf_ref", "path to high freq reference images",
            cxxopts::value<std::string>())("hf_obj",
                                           "path to high freq target images",
                                           cxxopts::value<std::string>())(
            "c, counter", "number of repetitions", cxxopts::value<int>(c))(
            "help", "print help");

        auto parse_res = opt.parse(argc, argv);

        if (parse_res.count("help")) {
            std::cout << opt.help() << std::endl;
            return 0;
        }

        m = parse_res["method"].as<std::string>();
        auto alg = sl_alg_make(m, {h, w});
        if (!alg) {
            std::cout << "method: " << m << " not supported";
            std::exit(1);
        }

        if (parse_res.count("rpc")) {
            // sla_grpc_srv_run(alg.get());
        } else {
            sla_run_once(sl_alg_type_by_name(m), alg.get(), parse_res, c);
        }

    } catch (const std::exception& e) {
        std::cout << e.what();
    }

    return 0;
}
