#include "lfs_transport.hpp"
#include "alg_utils.hpp"
#include <cstdint>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

void lfs_imgs_read(
    const ::google::protobuf::RepeatedPtrField<std::string>& file_paths,
    std::vector<cv::Mat>& outs) {
        for (auto i = 0; i < file_paths.size(); ++i) {
            outs[i] = cv::imread(file_paths.at(i), cv::IMREAD_GRAYSCALE);
        }
}
