#include "sl_pcg_alg.hpp"

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

class sl_pcg_alg::sl_alg_impl {
   public:
    sl_alg_impl(cv::Size size)
        : _temp_phases(cuda_imgs_alloc(4, size, CV_32F)),
          _hf_obj_phase(cuda_imgs_alloc(1, size, CV_32F)),
          _hf_ref_phase(cuda_imgs_alloc(1, size, CV_32F)),
          _cu_pu_alg(cuda_phase_unwrap_alg(size)),
          _filt(
              cv::cuda::createGaussianFilter(CV_32F, CV_32F, cv::Size(5, 5), 0))

    {}

    int ref_phase_compute(const std::vector<cv::cuda::GpuMat>& refs) {
        cuda_phase_compute(refs, _temp_phases, _hf_ref_phase[0], *_filt);
        return 0;
    }

    int obj_phase_compute(const std::vector<cv::cuda::GpuMat>& objs) {
        cuda_phase_compute(objs, _temp_phases, _hf_obj_phase[0], *_filt);
        cv::cuda::subtract(_hf_obj_phase[0], _hf_ref_phase[0], _hf_obj_phase[0]);

        return 0;
    }

    cv::Mat compute_3dr_impl() {
        return _cu_pu_alg.gradient_unwrap(_hf_obj_phase[0]);
    }

   private:
    std::vector<cv::cuda::GpuMat> _temp_phases;
    std::vector<cv::cuda::GpuMat> _hf_obj_phase;
    std::vector<cv::cuda::GpuMat> _hf_ref_phase;
    cuda_phase_unwrap_alg _cu_pu_alg;
    cv::Ptr<cv::cuda::Filter> _filt;
};

sl_pcg_alg::sl_pcg_alg(cv::Size size)
    : _pimpl(std::make_unique<sl_alg_impl>(size)) {}

sl_pcg_alg::~sl_pcg_alg() = default;

int sl_pcg_alg::ref_phase_compute(const std::vector<cv::cuda::GpuMat>& imgs) {
    return _pimpl->ref_phase_compute(imgs);
}

int sl_pcg_alg::obj_phase_compute(const std::vector<cv::cuda::GpuMat>& imgs) {
    return _pimpl->obj_phase_compute(imgs);
}

cv::Mat sl_pcg_alg::compute_3d_reconstruction() {
    return _pimpl->compute_3dr_impl();
}
