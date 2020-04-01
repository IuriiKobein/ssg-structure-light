#pragma once

#include <grpcpp/impl/codegen/service_type.h>
#include <memory>

std::unique_ptr<grpc::Service> sla_ctrl_make();
