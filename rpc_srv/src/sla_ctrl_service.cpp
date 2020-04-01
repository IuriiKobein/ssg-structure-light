#include <grpcpp/impl/codegen/status_code_enum.h>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>

#include <grpc/support/log.h>
#include <grpcpp/grpcpp.h>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "proto_gen/sla.grpc.pb.h"
#include "sl_alg_factory.hpp"

#include "alg_utils.hpp"
#include "lfs_transport.hpp"

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
const std::string LFS_PREFIX = "/tmp/";

std::string file_path_gen(const cv::Mat& img) {
    return LFS_PREFIX + "depth_map_" + std::to_string((std::intptr_t)img.data) +
           ".png";
}
}  // namespace

class sla_ctrl_impl final : public sla_ctrl::Service {
   public:
    sla_ctrl_impl(){};
    Status _setup(ServerContext* ctx, const conf_req* req,
                  status_res* res) override {
        const auto& method = req->method();

        sl_alg::params_t params;

        params.size = cv::Size(req->height(), req->width());
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

        res->set_status(0);

        return Status::OK;
    }

    Status _ref_phase_compute(ServerContext* ctx, const compute_req* req,
                              status_res* res) override {
        try {
            const auto& method = req->method();

            auto status = is_valid_method(method);
            if (!status.ok()) {
                return status;
            }

            auto rc = alg_ref_compute_invoke(method, req);
            if (rc) {
                return Status(grpc::StatusCode::INTERNAL, "internal");
            }

            res->set_status(0);

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

            auto status = is_valid_method(method);
            if (!status.ok()) {
                return status;
            }

            auto depth_mat = alg_depth_compute_invoke(method, req);
            auto lfs_url = file_path_gen(depth_mat);

            lfs_img_write(lfs_url, depth_mat);
            res->set_url_img(lfs_url);

            return Status::OK;
        } catch (const std::exception& e) {
            std::cout << e.what();
            return Status(grpc::StatusCode::INTERNAL, e.what());
        }
    }

   private:
    Status is_valid_method(const std::string& method) const {
        if (method.empty()) {
            return Status(grpc::StatusCode::INTERNAL, "method not specified");
        }

        if (sl_alg_is_supported(method)) {
            return Status(grpc::StatusCode::INTERNAL,
                          "method not supported " + method);
        }

        if (_alg_map.count(method)) {
            return Status(grpc::StatusCode::INTERNAL,
                          "method is not setup " + method);
        }

        return Status::OK;
    }

    int alg_ref_compute_invoke(const std::string& method,
                               const compute_req* req) {
        lfs_imgs_read(req->hf_img(), _hf_imgs);

        if (method.find("tpu") == std::string::npos) {
            return _alg_map[method]->ref_phase_compute(_hf_imgs);
        } else {
            lfs_imgs_read(req->lf_img(), _lf_imgs);
            return _alg_map[method]->ref_phase_compute(_lf_imgs, _hf_imgs);
        }
    }

    cv::Mat alg_depth_compute_invoke(const std::string& method,
                                     const compute_req* req) {
        lfs_imgs_read(req->hf_img(), _hf_imgs);

        if (method.find("tpu") == std::string::npos) {
            return _alg_map[method]->depth_compute(_hf_imgs);
        } else {
            lfs_imgs_read(req->lf_img(), _lf_imgs);
            return _alg_map[method]->depth_compute(_lf_imgs, _hf_imgs);
        }
    }

    sl_alg::params_t _params;
    std::vector<cv::Mat> _lf_imgs;
    std::vector<cv::Mat> _hf_imgs;
    std::unordered_map<std::string, std::unique_ptr<sl_alg>> _alg_map;
};

std::unique_ptr<grpc::Service> sla_ctrl_make() {
    return std::make_unique<sla_ctrl_impl>();
}
