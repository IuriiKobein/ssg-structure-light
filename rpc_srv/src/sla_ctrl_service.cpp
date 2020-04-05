#include <grpcpp/impl/codegen/status_code_enum.h>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>

#include <grpc/support/log.h>
#include <grpcpp/grpcpp.h>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "proto_gen/sla.grpc.pb.h"
#include "sl_alg_factory.hpp"

#include "alg_utils.hpp"
#include "lfs_transport.hpp"
#include "proj_cam_srv.hpp"

using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerCompletionQueue;
using grpc::ServerContext;
using grpc::Status;

using sla::compute_req;
using sla::compute_res;
using sla::conf_req;
using sla::scan_req;
using sla::sla_ctrl;
using sla::status_res;

namespace {
using img_vec_t = std::vector<cv::Mat>;
using alg_map_t = std::unordered_map<std::string, std::unique_ptr<sl_alg>>;

const std::string LFS_PREFIX = "/tmp/";

std::string file_path_gen(const cv::Mat& img) {
    return LFS_PREFIX + "depth_map_" + std::to_string((std::intptr_t)img.data) +
           ".png";
}

Status req_imgs_load(const compute_req* req, img_vec_t& lf_imgs,
                     img_vec_t& hf_imgs) {
    auto rc = lfs_imgs_read(req->hf_img(), hf_imgs);
    if (rc) {
        return Status(grpc::StatusCode::INTERNAL,
                      "failed to load hf images: " + std::to_string(rc));
    }

    rc = lfs_imgs_read(req->lf_img(), lf_imgs);
    if (rc) {
        return Status(grpc::StatusCode::INTERNAL,
                      "failed to load lf images: " + std::to_string(rc));
    }

    return Status::OK;
}

Status is_valid_method(const alg_map_t& alg_map, const std::string& method) {
    if (method.empty()) {
        return Status(grpc::StatusCode::INTERNAL, "method not specified");
    }
    if (!sl_alg_is_supported(method)) {
        return Status(grpc::StatusCode::INTERNAL,
                      "method not supported " + method);
    }

    if (!alg_map.count(method)) {
        return Status(grpc::StatusCode::INTERNAL,
                      "method is not setup " + method);
    }

    return Status::OK;
}

Status alg_ref_compute_invoke(const alg_map_t& alg_map,
                              const std::string& method,
                              const img_vec_t& lf_imgs,
                              const img_vec_t& hf_imgs,
                              cv::Mat& out) {
    if (method.find("tpu") == std::string::npos) {
        out = alg_map.at(method)->ref_phase_compute(hf_imgs);
    } else {
        out = alg_map.at(method)->ref_phase_compute(lf_imgs, hf_imgs);
    }

    if (out.empty()) {
        return Status(
            grpc::StatusCode::INTERNAL,
            "failed to compute ref phase: " + method);
    }

    return Status::OK;
}

Status alg_depth_compute_invoke(const alg_map_t& alg_map,
                                const std::string& method,
                                const img_vec_t& lf_imgs,
                                const img_vec_t& hf_imgs, cv::Mat& out) {
    if (method.find("tpu") == std::string::npos) {
        out = alg_map.at(method)->depth_compute(hf_imgs);
    } else {
        out = alg_map.at(method)->depth_compute(lf_imgs, hf_imgs);
    }

    return Status::OK;
}


Status alg_proj_and_capture(const alg_map_t& alg_map, const std::string& method,
                            proj_cam_srv& proj_cam, img_vec_t& lf_imgs,
                            img_vec_t& hf_imgs) {
    const auto& patterns = alg_map.at(method)->patterns_get();

    if (method.find("tpu") != std::string::npos) {
        proj_cam.images_capture(std::begin(patterns),
                                    std::end(patterns) - patterns.size() / 2,
                                    std::begin(lf_imgs));
        proj_cam.images_capture(std::begin(patterns) + patterns.size() / 2,
                                    std::end(patterns), std::begin(hf_imgs));
    } else {
        proj_cam.images_capture(std::begin(patterns), std::end(patterns),
                                    std::begin(hf_imgs));
    }

    return Status::OK;
}
}  // namespace

class sla_ctrl_impl final : public sla_ctrl::Service {
   public:
    sla_ctrl_impl(){};
    Status _setup(ServerContext* ctx, const conf_req* req,
                  status_res* res) override {
        const auto& method = req->method();

        sl_alg::params_t params;

        params.size = cv::Size(req->width(), req->height());
        params.freq_ratio = req->freq_ratio();
        params.is_horizontal = req->is_horizontal();
        params.num_of_periods = req->num_of_periods();
        params.num_of_pix_mark = req->num_of_pix_mark();
        params.opencv_method_id = req->opencv_method_id();
        params.real_scale = req->real_scale();
        params.num_of_patterns = req->num_of_patterns();
        params.use_markers = req->use_markers();

        _alg_map[method] = sl_alg_make(method, params);

        _lf_imgs = imgs_alloc(4, params.size, CV_8UC1);
        _hf_imgs = imgs_alloc(4, params.size, CV_8UC1);

        _proj_cam.size_set(params.size);

        std::cout << "setup method:  " << method << " :"
                  << std::to_string(req->opencv_method_id());
        res->set_status(0);

        return Status::OK;
    }

    Status _ref_phase_compute(ServerContext* ctx, const compute_req* req,
                              compute_res* res) override {
        try {
            const auto& method = req->method();

            auto status = is_valid_method(_alg_map, method);
            if (!status.ok()) {
                res->set_url_img(std::string("error") + std::to_string(status.error_code()));
                return status;
            }

            status = req_imgs_load(req, _lf_imgs, _hf_imgs);
            if (!status.ok()) {
                res->set_url_img(std::string("error") + std::to_string(status.error_code()));
                return status;
            }

            cv::Mat ref;
            status =
                alg_ref_compute_invoke(_alg_map, method, _lf_imgs, _hf_imgs, ref);
            if (!status.ok()) {
                res->set_url_img(std::string("error") + std::to_string(status.error_code()));
                return status;
            }

            auto lfs_url = file_path_gen(ref);
            lfs_img_write(lfs_url, ref);
            res->set_url_img(lfs_url);

            return Status::OK;
        } catch (const std::exception& e) {
            std::cout << e.what();
            return Status(grpc::StatusCode::INTERNAL, e.what());
        }
    }

    Status _depth_compute(ServerContext* ctx, const compute_req* req,
                          compute_res* res) override {
        try {
            const auto& method = req->method();

            auto status = is_valid_method(_alg_map, method);
            if (!status.ok()) {
                return status;
            }

            status = alg_depth_compute_invoke(_alg_map, method, _lf_imgs,
                                              _hf_imgs, _depth_map);
            if (!status.ok()) {
                return status;
            }

            auto lfs_url = file_path_gen(_depth_map);
            lfs_img_write(lfs_url, _depth_map);
            res->set_url_img(lfs_url);

            return Status::OK;
        } catch (const std::exception& e) {
            std::cout << e.what();
            return Status(grpc::StatusCode::INTERNAL, e.what());
        }
    }

    Status _ref_phase_capture_and_compute(ServerContext* ctx,
                                          const compute_req* req,
                                          compute_res* res) override {
        try {
            const auto& method = req->method();

            auto status = is_valid_method(_alg_map, method);
            if (!status.ok()) {
                res->set_url_img(std::string("error") + std::to_string(status.error_code()));
                return status;
            }

            status = alg_proj_and_capture(_alg_map, method, _proj_cam, _lf_imgs,
                                          _hf_imgs);
            if (!status.ok()) {
                res->set_url_img(std::string("error") + std::to_string(status.error_code()));
                return status;
            }

            cv::Mat ref;
            status =
                alg_ref_compute_invoke(_alg_map, method, _lf_imgs, _hf_imgs, ref);
            if (!status.ok()) {
                res->set_url_img(std::string("error") + std::to_string(status.error_code()));
                return status;
            }

            auto lfs_url = file_path_gen(ref);
            lfs_img_write(lfs_url, ref);
            res->set_url_img(lfs_url);



            return Status::OK;
        } catch (const std::exception& e) {
            std::cout << e.what();
            return Status(grpc::StatusCode::INTERNAL, e.what());
        }
    }

    Status _depth_capture_and_compute(ServerContext* ctx,
                                      const compute_req* req,
                                      compute_res* res) override {
        try {
            const auto& method = req->method();

            auto status = is_valid_method(_alg_map, method);
            if (!status.ok()) {
                return status;
            }

            status = alg_proj_and_capture(_alg_map, method, _proj_cam, _lf_imgs,
                                          _hf_imgs);
            if (!status.ok()) {
                return status;
            }

            status = alg_depth_compute_invoke(_alg_map, method, _lf_imgs,
                                              _hf_imgs, _depth_map);
            if (!status.ok()) {
                return status;
            }

            auto lfs_url = file_path_gen(_depth_map);
            lfs_img_write(lfs_url, _depth_map);
            res->set_url_img(lfs_url);

            return Status::OK;
        } catch (const std::exception& e) {
            std::cout << e.what();
            return Status(grpc::StatusCode::INTERNAL, e.what());
        }
    }

   private:
    proj_cam_srv _proj_cam;
    sl_alg::params_t _params;
    std::unordered_map<std::string, std::unique_ptr<sl_alg>> _alg_map;
    std::vector<cv::Mat> _lf_imgs;
    std::vector<cv::Mat> _hf_imgs;
    cv::Mat _depth_map;
};

std::unique_ptr<grpc::Service> sla_ctrl_make() {
    return std::make_unique<sla_ctrl_impl>();
}
