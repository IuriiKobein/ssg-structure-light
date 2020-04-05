#include "lfs_transport.hpp"
#include <asm-generic/errno-base.h>
#include <cstdint>
#include <exception>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include "alg_utils.hpp"

int lfs_imgs_read(
    const ::google::protobuf::RepeatedPtrField<std::string>& file_paths,
    std::vector<cv::Mat>& outs) {
    try {
        for (auto i = 0; i < file_paths.size(); ++i) {
            auto orig = cv::imread(file_paths.at(i), cv::IMREAD_GRAYSCALE);
            cv::resize(orig, outs[i], outs[i].size());
        }
    } catch (const std::exception& e) {
        std::cout << e.what() << '\n';
        outs.clear();
        return -EINVAL;
    }

    return 0;
}
