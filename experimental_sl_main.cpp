#include <exception>
#include <iostream>

#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <string>

#include "alg_utils.hpp"
#include "rpc_server.hpp"
#include "sl_alg_factory.hpp"

namespace {

const char* keys = {
    "{mode | rpc | Run mode : 'rpc', 'capture_compute','compute'}"
    "{method || Method to be used: '_tpu', '_pcg'}"
    "{width | -1 | Projector width}"
    "{height | -1 | Projector height}"
    "{tpu_fr_ratio | 20 | hf/lf ration for TPU}"
    "{num_of_periods | 40 | Number of periods in pattern}"
    "{num_of_patterns | 4 | Number of patterns}"
    "{horizontal |1| Patterns are horizontal}"
    "{lf_ref_path |<none>| Path to captured lf ref images}"
    "{hf_ref_path |<none>| Path to captured hf ref images}"
    "{lf_obj_path |<none>| Path to captured lf obj images}"
    "{hf_obj_path |<none>| Path to captured hf obj images}"
    "{outputPatternPath | | Path to save patterns}"
    "{outputWrappedPhasePath | | Path to save wrapped phase map}"
    "{outputUnwrappedPhasePath | | Path to save unwrapped phase map}"
    "{outputCapturePath | | Path to save the captures}"
    "{reliabilitiesPath | | Path to save reliabilities}"
    "{rpc_address | localhost:12321 | RPC server address}"};

void alg_3dr_tpu_ref_upload(sl_alg& alg, const std::string& lf_path,
                            const std::string& hf_path) {
    auto lf_ref = imgs_from_dir_load(lf_path);
    auto hf_ref = imgs_from_dir_load(hf_path);

    alg.ref_phase_compute(lf_ref, hf_ref);
}

cv::Mat alg_3dr_tpu_obj_depth_compute(sl_alg& alg, const std::string& lf_path,
                                      const std::string& hf_path) {
    auto lf_obj = imgs_from_dir_load(lf_path);
    auto hf_obj = imgs_from_dir_load(hf_path);

    return alg.depth_compute(lf_obj, hf_obj);
}

void alg_3dr_pcg_ref_upload(sl_alg& alg, const std::string& hf_path) {
    auto hf_img = imgs_from_dir_load(hf_path);

    alg.ref_phase_compute(hf_img);
}

cv::Mat alg_3dr_pcg_obj_depth_compute(sl_alg& alg, const std::string& hf_path) {
    auto hf_obj = imgs_from_dir_load(hf_path);

    return alg.depth_compute(hf_obj);
}

void alg_3dr_imgs_ref_compute(sl_alg& alg, cv::CommandLineParser& parser,
                              int type) {
    if (type == TPU) {
        alg_3dr_tpu_ref_upload(alg, parser.get<cv::String>("lf_ref_path"),
                               parser.get<cv::String>("hf_ref_path"));
    } else if (type == PCG) {
        alg_3dr_pcg_ref_upload(alg, parser.get<cv::String>("hf_ref_path"));
    }
}

cv::Mat alg_3dr_obj_depth_compute(sl_alg& alg, cv::CommandLineParser& parser,
                                  int type) {
    if (type == TPU) {
        return alg_3dr_tpu_obj_depth_compute(
            alg, parser.get<cv::String>("lf_obj_path"),
            parser.get<cv::String>("hf_obj_path"));
    } else if (type == PCG) {
        return alg_3dr_pcg_obj_depth_compute(
            alg, parser.get<cv::String>("hf_obj_path"));
    }

    return cv::Mat();
}

int sl_run_once(cv::CommandLineParser& parser) {
    auto m = parser.get<cv::String>("method");
    auto w = parser.get<int>("width");
    auto h = parser.get<int>("height");
    auto freq_ratio = parser.get<std::float_t>("tpu_fr_ratio");
    auto alg_type = sl_alg_type_by_name(m);

    sl_alg::params_t params;

    params.size = cv::Size(h, w);
    params.freq_ratio = freq_ratio;
    params.is_horizontal = 0;
    params.num_of_periods = 32;
    params.num_of_pix_mark = -1;
    params.opencv_method_id = -1;
    params.real_scale = 1;
    params.num_of_patterns = 4;
    params.use_markers = 0;

    auto alg = sl_alg_make(m, params);
    auto patterns = alg->patterns_get();

    img_show("p1", patterns[0]);
    cv::waitKey(0);
    if (!alg) {
        std::cout << "method: " << m << " not supported";
        std::exit(1);
    }

    alg_3dr_imgs_ref_compute(*alg, parser, alg_type);

    auto out = alg_3dr_obj_depth_compute(*alg, parser, alg_type);

    auto ts = std::chrono::high_resolution_clock::now();
    out = alg_3dr_obj_depth_compute(*alg, parser, alg_type);
    auto te = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(te - ts)
                     .count()
              << std::endl;

    img_show("cuda", out);
    cv::waitKey(0);

    return 0;
}
}  // namespace

int experimental_sl_main(int argc, char* argv[]) {
    cv::CommandLineParser parser(argc, argv, keys);

    try {
        auto mode = parser.get<std::string>("mode");
        if (mode == "rpc") {
            rpc_server_run(parser.get<std::string>("rpc_address"));
        } else {
            sl_run_once(parser);
        }
    } catch (const std::exception& e) {
        std::cout << e.what();
    }

    return 0;
}
