#pragma once

#include "sl_alg.hpp"

class sl_pcg : public sl_alg {
   public:
    sl_pcg(cv::Size size);
    ~sl_pcg();

    virtual int ref_phase_compute(const std::vector<cv::Mat> &ref_phases);
    virtual cv::Mat depth_compute(const std::vector<cv::Mat> &obj_phases);

   private:
    class sl_pcg_impl;
    std::unique_ptr<sl_pcg_impl> _pimpl;
};

