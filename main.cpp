#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
#include <memory>

#include <opencv2/core/utility.hpp>
#include <string>
#include <type_traits>
#include <vector>

#include "experimental_sl_main.hpp"
#include "opencv_sl_main.hpp"

namespace {

const char* keys = {"{sla_impl | experimental | Implementation of SLA}"};
}

int main(int argc, char* argv[]) {
    cv::CommandLineParser parser(argc, argv, keys);

    auto impl = parser.get<std::string>("sla_impl");

    if (impl == "experimental") {
        experimental_sl_main(argc, argv);
    } else if (impl == "opencv") {
        // exclude first argument not to confuse underlaying modules
        for (int i = 1; i < argc; ++i) argv[i] = argv[i + 1];
        --argc;

        opencv_sl_main(argc, argv);
    }
    return 0;
}
