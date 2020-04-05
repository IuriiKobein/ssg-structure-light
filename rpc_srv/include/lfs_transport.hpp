#pragma once

#include <google/protobuf/api.pb.h>
#include <opencv2/core.hpp>

int lfs_imgs_read(
    const ::google::protobuf::RepeatedPtrField<std::string>& file_paths,
    std::vector<cv::Mat>& outs);
