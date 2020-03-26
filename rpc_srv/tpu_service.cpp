#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include <grpc/support/log.h>
#include <grpcpp/grpcpp.h>
#include <unordered_map>

#include "proto_gen/sla.grpc.pb.h"
#include "sl_alg.hpp"

using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerCompletionQueue;
using grpc::ServerContext;
using grpc::Status;

using sla::tpu;
using sla::tpu_conf_req;
using sla::tpu_conf_res;
using sla::tpu_calc_req;
using sla::tpu_calc_res;

class tpu_srv_impl final : public tpu::Service {
   public:
    tpu_srv_impl(){};
    Status _init(ServerContext* ctx, const tpu_conf_req* req,
                 tpu_conf_res* res) override {
        
        return Status::OK;
    }

   private:
    std::unique_ptr<sl_alg> _tpu;
};
