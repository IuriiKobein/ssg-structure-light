#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <string> 
#include "structs.h"

void phaseUnwrap(cv::cuda::GpuMat &img, ConstData &constGrids, VarMats &varMats);
