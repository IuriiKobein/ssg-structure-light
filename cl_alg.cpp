#include "cl_alg.hpp"

#include "alg_utils.hpp"

#include "cuda_kernels.h"
#include "cuda_phase_unwrap.hpp"

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/highgui.hpp>

#include <string>
#include <type_traits>

namespace {
void to_float_scale(const std::vector<cv::cuda::GpuMat>& src,
                    std::vector<cv::cuda::GpuMat>& dst) {
    for (auto i = 0ul; i < src.size(); ++i) {
        src[i].convertTo(dst[i], dst[i].type(), 1.0f / 255);
    }
}

void cuda_phase_compute(const std::vector<cv::cuda::GpuMat>& src,
                        std::vector<cv::cuda::GpuMat>& dst,
                        cv::cuda::GpuMat& out) {
    to_float_scale(src, dst);
    cuda_diff_atan_inplace(dst, out);
}
}  // namespace

class structure_light_alg::sl_alg_impl {
   public:
    sl_alg_impl(cv::Size size)
        : _phases(cuda_imgs_alloc(4, size, CV_32F)),
          _src_phase(cv::cuda::GpuMat(size, CV_32F)),
          _ref_phase(cv::cuda::GpuMat(size, CV_32F)),
          _cu_pu_alg(cuda_phase_unwrap_alg(size))

    {}

    int ref_phases_compute(const std::vector<cv::cuda::GpuMat>& imgs) { 
        cuda_phase_compute(imgs, _phases, _ref_phase);
        return 0;
    }

    cv::Mat compute_3dr_impl(const std::vector<cv::cuda::GpuMat>& imgs) {
        // auto filt =
        //    cv::cuda::createGaussianFilter(CV_32F, CV_32F, cv::Size(5, 5), 0);
        // std::for_each(std::begin(srcs_f32), std::end(srcs_f32),
        //              [&filt](auto &img) { filt->apply(img, img); });

        cuda_phase_compute(imgs, _phases, _src_phase);

        cv::cuda::subtract(_src_phase, _ref_phase, _src_phase);

        return _cu_pu_alg.compute(_src_phase);
    }

   private:
    std::vector<cv::cuda::GpuMat> _phases;
    cv::cuda::GpuMat _src_phase;
    cv::cuda::GpuMat _ref_phase;
    cuda_phase_unwrap_alg _cu_pu_alg;
};

structure_light_alg::structure_light_alg(cv::Size size)
    : _pimpl(std::make_unique<sl_alg_impl>(size)) {}

structure_light_alg::~structure_light_alg() = default;

int structure_light_alg::ref_phase_compute(
    const std::vector<cv::cuda::GpuMat>& imgs) {
    return _pimpl->ref_phases_compute(imgs);
}

cv::Mat structure_light_alg::compute_3d_reconstruction(
    const std::vector<cv::cuda::GpuMat>& imgs) {
    return _pimpl->compute_3dr_impl(imgs);
}
