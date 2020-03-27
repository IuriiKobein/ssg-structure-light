#include "sl_tpu_alg.hpp"
#include "alg_utils.hpp"

#include <algorithm>
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
    alg_impl(cv::Size size, tpu_params_t params)
        : _temp_phases(cuda_imgs_alloc(4, size, CV_32F)),
          _lf_obj_phase(cuda_img_alloc(size, CV_32F)),
          _hf_obj_phase(cuda_img_alloc(size, CV_32F)),
          _lf_ref_phase(cuda_img_alloc(size, CV_32F)),
          _hf_ref_phase(cuda_img_alloc(size, CV_32F)),
          _cu_pu_alg(cuda_phase_unwrap_alg(size, params)),
          _filt(
              cv::cuda::createGaussianFilter(CV_32F, CV_32F, cv::Size(5, 5), 0))

    {}

    int ref_phase_compute(const std::vector<cv::cuda::GpuMat>& refs) {
        auto it = std::begin(refs);

        cuda_phase_compute(it, (it + 4), _temp_phases, *_filt, _lf_ref_phase);
        cuda_phase_compute((it+4), (it + 8), _temp_phases, *_filt, _hf_ref_phase);

        return 0;
    }

    int obj_phase_compute(const std::vector<cv::cuda::GpuMat>& objs) {
        auto it = std::begin(objs);

        cuda_phase_compute(it, it + 4, _temp_phases, *_filt, _lf_obj_phase);
        cuda_phase_compute(it + 4, it + 8, _temp_phases, *_filt, _hf_obj_phase);

        cv::cuda::subtract(_lf_obj_phase, _lf_ref_phase, _lf_obj_phase);
        cv::cuda::subtract(_hf_obj_phase, _hf_ref_phase, _hf_obj_phase);

        return 0;
    }

    cv::Mat compute_3dr_impl() {
        return _cu_pu_alg.temporal_unwrap(_lf_obj_phase, _hf_obj_phase, 20);
    }

   private:
    std::vector<cv::cuda::GpuMat> _temp_phases;
    cv::cuda::GpuMat _lf_obj_phase;
    cv::cuda::GpuMat _hf_obj_phase;
    cv::cuda::GpuMat _lf_ref_phase;
    cv::cuda::GpuMat _hf_ref_phase;
    cv::Ptr<cv::cuda::Filter> _filt;
};

sl_tpu_alg::sl_tpu_alg(cv::Size size)
    : _pimpl(std::make_unique<alg_impl>(size)) {}

sl_tpu_alg::~sl_tpu_alg() = default;

int sl_tpu_alg::ref_phase_compute(const std::vector<cv::cuda::GpuMat>& imgs) {
    return -ENOTSUP;
}

int sl_tpu_alg::obj_phase_compute(const std::vector<cv::cuda::GpuMat>& imgs) {
    return _pimpl->obj_phase_compute(imgs);
}
