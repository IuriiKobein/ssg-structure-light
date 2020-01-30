#include "cl_alg.hpp"

#include "alg_utils.hpp"

#include "cuda_kernels.h"
#include "cuda_phase_unwrap.hpp"

#include <algorithm>
#include <iterator>
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
                        cv::cuda::GpuMat& out, cv::Ptr<cv::cuda::Filter> filt) {
    to_float_scale(src, dst);
    std::for_each(std::begin(dst), std::end(dst), [&filt](auto& i) {
        filt->apply(i, i);});
    cuda_diff_atan_inplace(dst, out);
}
}  // namespace

class structure_light_alg::sl_alg_impl {
   public:
    sl_alg_impl(cv::Size size, float flag)
        : _temp_phases(cuda_imgs_alloc(4, size, CV_32F)),
          _src_phase(cuda_imgs_alloc_type(flag, size, CV_32F)),
          _ref_phase(cuda_imgs_alloc_type(flag, size, CV_32F)),
          _cu_pu_alg(cuda_phase_unwrap_alg(size)),
          _filt(
              cv::cuda::createGaussianFilter(CV_32F, CV_32F, cv::Size(5, 5), 0))

    {}

    int ref_phases_compute(const std::vector<cv::cuda::GpuMat>& imgs, int i) {

        cuda_phase_compute(imgs, _temp_phases, _ref_phase[i], _filt);
        return 0;
    }

    int obj_phases_compute(const std::vector<cv::cuda::GpuMat>& imgs, int i) {

        cuda_phase_compute(imgs, _temp_phases, _src_phase[i], _filt);
        cv::cuda::subtract(_src_phase[i], _ref_phase[i], _src_phase[i]);
        return 0;
    }

    cv::Mat compute_3dr_impl(int mode) {

        if(mode == 1){
            return _cu_pu_alg.gradient_unwrap(_src_phase[0]);
        }else{
            return _cu_pu_alg.temporal_unwrap(_src_phase[0], _src_phase[1], 20);
        }
    }

   private:
    std::vector<cv::cuda::GpuMat> _temp_phases;
    std::vector<cv::cuda::GpuMat> _src_phase;
    std::vector<cv::cuda::GpuMat> _ref_phase;
    cuda_phase_unwrap_alg _cu_pu_alg;
    cv::Ptr<cv::cuda::Filter> _filt;
};

structure_light_alg::structure_light_alg(cv::Size size, int flag)
    : _pimpl(std::make_unique<sl_alg_impl>(size, flag)) {}

structure_light_alg::~structure_light_alg() = default;

int structure_light_alg::ref_phase_compute(
    const std::vector<cv::cuda::GpuMat>& imgs, int i) {
    return _pimpl->ref_phases_compute(imgs, i);
}

int structure_light_alg::obj_phase_compute(
    const std::vector<cv::cuda::GpuMat>& imgs, int i) {
    return _pimpl->obj_phases_compute(imgs, i);
}

cv::Mat structure_light_alg::compute_3d_reconstruction(int mode) {
    return _pimpl->compute_3dr_impl(mode);
}
