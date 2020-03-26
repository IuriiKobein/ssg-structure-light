#pragma once

#include "sl_alg.hpp"

class sl_pcg : public sl_alg {
   public:
    sl_pcg(cv::Size size);
    ~sl_pcg();

    virtual int ref_phase_compute(const std::vector<cv::Mat> &ref_phases);
    virtual cv::Mat depth_compute(const std::vector<cv::Mat> &obj_phases);
    virtual int ref_phase_compute(const std::vector<cv::Mat> &lf_refs,
            const std::vector<cv::Mat>& hf_refs);
    virtual cv::Mat depth_compute(const std::vector<cv::Mat> &lf_objs,
            const std::vector<cv::Mat> &hf_objs);

   private:
    class sl_pcg_impl;
    std::unique_ptr<sl_pcg_impl> _pimpl;
};

