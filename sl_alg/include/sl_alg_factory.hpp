#pragma once

#include <functional>
#include <memory>
#include <opencv2/core/types.hpp>
#include <string>
#include "sl_alg.hpp"

enum alg_type { PCG = 1, TPU = 2 };
using sl_alg_maker_fn = std::function<std::unique_ptr<sl_alg>(const sl_alg::params_t&)>;

alg_type sl_alg_type_by_name(const std::string& name);
int sl_factory_alg_register(const std::string& name, sl_alg_maker_fn maker);
std::unique_ptr<sl_alg> sl_alg_make(const std::string& name, const sl_alg::params_t& params);

struct sl_alg_auto_reg {
    sl_alg_auto_reg(const std::string& name, sl_alg_maker_fn maker)
    {
        sl_factory_alg_register(name, std::move(maker));
    }
};


