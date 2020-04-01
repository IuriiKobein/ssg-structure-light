#include "sl_alg_factory.hpp"

#include <functional>
#include <memory>
#include <opencv2/core/types.hpp>
#include <unordered_map>

#include "cu_sl_pcg.hpp"
#include "sl_pcg.hpp"

namespace {
auto& registrar_map_get() {
    static std::unordered_map<std::string, sl_alg_maker_fn> s_registrar;

    return s_registrar;
}
}  // namespace

int sl_alg_is_supported(const std::string& name) {
    return registrar_map_get().count(name);
}

alg_type sl_alg_type_by_name(const std::string& name) {
    if (name == "cuda_pcg" || name == "cpu_pcg") {
        return PCG;
    }

    return TPU;
}

int sl_factory_alg_register(const std::string& name, sl_alg_maker_fn maker) {
    auto& reg = registrar_map_get();

    if (reg.count(name)) {
        return -EEXIST;
    }

    reg[name] = std::move(maker);

    return 0;
}

std::unique_ptr<sl_alg> sl_alg_make(const std::string& name,
                                    const sl_alg::params_t& params) {
    auto& reg = registrar_map_get();

    if (!reg.count(name)) {
        return nullptr;
    }

    return reg[name](params);
}
