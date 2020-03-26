#pragma once

#include "sl_alg.hpp"

class cu_sl_pcg : public sl_alg {
   public:
    cu_sl_pcg(cv::Size size);
    ~cu_sl_pcg();

    virtual int ref_phase_compute(const std::vector<cv::Mat> &refs);
    virtual cv::Mat depth_compute(const std::vector<cv::Mat> &objs);

   private:
    class cu_sl_pcg_impl;
    std::unique_ptr<cu_sl_pcg_impl> _pimpl;
};

