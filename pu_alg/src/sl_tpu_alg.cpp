#include "sl_tpu_alg.hpp"

#include "alg_utils.hpp"

#include "cuda_kernels.h"
#include "cuda_phase_unwrap.hpp"

#include <c++/7/bits/c++config.h>
#include <algorithm>
#include <iterator>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/highgui.hpp>

#include <string>
#include <type_traits>
#include <vector>

namespace {
template <typename Iter>
void cuda_phase_compute(Iter begin, Iter end,
                        std::vector<cv::cuda::GpuMat>& temp,
                        cv::cuda::Filter& filt, cv::cuda::GpuMat& out) {
    std::size_t idx = 0;

    std::for_each(begin, end, [temp, &filt, &idx](auto& img) {
        img.convertTo(temp[idx], temp[idx].type(), 1.0f / 255);
        filt.apply(temp[idx], temp[idx]);
        ++idx;
    });

    cuda_diff_atan_inplace(temp, out);
}
}  // namespace

class sl_tpu_alg::alg_impl {
   public:
    alg_impl(cv::Size size)
        : _temp_phases(cuda_imgs_alloc(4, size, CV_32F)),
          _lf_obj_phase(cuda_imgs_alloc(1, size, CV_32F)),
          _hf_obj_phase(cuda_imgs_alloc(1, size, CV_32F)),
          _lf_ref_phase(cuda_imgs_alloc(1, size, CV_32F)),
          _hf_ref_phase(cuda_imgs_alloc(1, size, CV_32F)),
          _cu_pu_alg(cuda_phase_unwrap_alg(size)),
          _filt(
              cv::cuda::createGaussianFilter(CV_32F, CV_32F, cv::Size(5, 5), 0))

    {}

    int ref_phase_compute(const std::vector<cv::cuda::GpuMat>& hf_refs) {
        auto it = std::begin(hf_refs);

        cuda_phase_compute(it, (it + 4), _temp_phases, *_filt, _lf_ref_phase[0]);
        cuda_phase_compute((it+4), (it + 8), _temp_phases, *_filt, _hf_ref_phase[0]);

        return 0;
    }

    int obj_phase_compute(const std::vector<cv::cuda::GpuMat>& hf_objs) {
        auto it = std::begin(hf_objs);

        cuda_phase_compute(it, (it + 4), _temp_phases, *_filt, _lf_ref_phase[0]);
        cv::cuda::subtract(_hf_obj_phase, _hf_ref_phase, _hf_obj_phase[0]);

        cuda_phase_compute((it+4), (it + 8), _temp_phases, *_filt, _hf_ref_phase[0]);
        cv::cuda::subtract(_hf_obj_phase, _hf_ref_phase, _hf_obj_phase[0]);
        return 0;
    }

    cv::Mat compute_3dr_impl() {
        return _cu_pu_alg.temporal_unwrap(_lf_obj_phase[0], _hf_obj_phase[0], 20);
    }

   private:
    std::vector<cv::cuda::GpuMat> _temp_phases;
    std::vector<cv::cuda::GpuMat> _lf_obj_phase;
    std::vector<cv::cuda::GpuMat> _hf_obj_phase;
    std::vector<cv::cuda::GpuMat> _lf_ref_phase;
    std::vector<cv::cuda::GpuMat> _hf_ref_phase;
    cuda_phase_unwrap_alg _cu_pu_alg;
    cv::Ptr<cv::cuda::Filter> _filt;
};

sl_tpu_alg::sl_tpu_alg(cv::Size size)
    : _pimpl(std::make_unique<alg_impl>(size)) {}

sl_tpu_alg::~sl_tpu_alg() = default;

int sl_tpu_alg::ref_phase_compute(const std::vector<cv::cuda::GpuMat>& imgs) {
    return _pimpl->ref_phase_compute(imgs);
}

int sl_tpu_alg::obj_phase_compute(const std::vector<cv::cuda::GpuMat>& imgs) {
    return _pimpl->obj_phase_compute(imgs);
}

cv::Mat sl_tpu_alg::compute_3d_reconstruction() {
    return _pimpl->compute_3dr_impl();
}
