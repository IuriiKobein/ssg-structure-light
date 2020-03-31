#include <grpcpp/impl/codegen/status_code_enum.h>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>

#include <grpc/support/log.h>
#include <grpcpp/grpcpp.h>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
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

using sla::depth_env_res;
using sla::tpu;
using sla::tpu_conf_req;
using sla::tpu_env_req;
using sla::tpu_status_res;

static const std::string LFS_PREFIX = "/tmp/";

class tpu_srv_impl final : public tpu::Service {
   public:
    tpu_srv_impl(){};
    Status _init(ServerContext* ctx, const tpu_conf_req* req,
                 tpu_status_res* res) override {
        auto method_impl = req->impl();
        auto size = cv::Size(req->height(), req->width());
        auto freq_ratio = req->freq_ratio();

        std::cout << "tpu init req\n";

        _tpu =
            sl_alg_make(method_impl, sl_alg::params_t{size, freq_ratio, 1.0f});

        _lf_imgs = imgs_alloc(4, size, CV_8UC1);
        _hf_imgs = imgs_alloc(4, size, CV_8UC1);

        res->set_status(0);

        return Status::OK;
    }

    Status _ref_phase_compute(ServerContext* ctx, const tpu_env_req* req,
                              tpu_status_res* res) override {
        try {
            if ((req->hf_img_size() != req->lf_img_size()) ||
                req->hf_img_size() != _lf_imgs.size()) {
                return Status(grpc::StatusCode::OUT_OF_RANGE,
                              "incorrect num of images");
            }

            lfs_imgs_read(req->lf_img(), _lf_imgs);
            lfs_imgs_read(req->hf_img(), _hf_imgs);

            if (_tpu->ref_phase_compute(_lf_imgs, _hf_imgs)) {
                return Status(grpc::StatusCode::INTERNAL, "internal");
            }

            res->set_status(0);

            return Status::OK;

        } catch (const std::exception& e) {
            std::cout << e.what();
            return Status(grpc::StatusCode::INTERNAL, e.what());
        }
    }

    Status _depth_compute(ServerContext* ctx, const tpu_env_req* req,
                          depth_env_res* res) override {
        try {
            if ((req->hf_img_size() != req->lf_img_size()) ||
                req->hf_img_size() != _lf_imgs.size()) {
                return Status(grpc::StatusCode::OUT_OF_RANGE,
                              "incorrect num of images");
            }

            lfs_imgs_read(req->lf_img(), _lf_imgs);
            lfs_imgs_read(req->hf_img(), _hf_imgs);

            auto depth_mat = _tpu->depth_compute(_lf_imgs, _hf_imgs);

            auto lfs_url = LFS_PREFIX + "depth_map_" +
                           std::to_string((std::intptr_t)depth_mat.data) +
                           ".png";
            lfs_img_write(lfs_url, depth_mat);
            res->set_url_unwrap_phase(lfs_url);

            return Status::OK;
        } catch (const std::exception& e) {
            std::cout << e.what();
            return Status(grpc::StatusCode::INTERNAL, e.what());
        }
    }

   private:
    sl_alg::params_t _params;
    std::vector<cv::Mat> _lf_imgs;
    std::vector<cv::Mat> _hf_imgs;
    std::unique_ptr<sl_alg> _tpu;
};

std::unique_ptr<grpc::Service> tpu_srv_make() {
    return std::make_unique<tpu_srv_impl>();
}
