# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the GRPC helloworld.Greeter client."""

from __future__ import print_function
import logging

import grpc

import sla_pb2
import sla_pb2_grpc


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('localhost:12321') as channel:
        print(channel)
        stub = sla_pb2_grpc.tpuStub(channel)
        print(stub)

        response = stub._init(sla_pb2.tpu_conf_req(impl="cuda_tpu",width=1024,height=1024,freq_ratio=20,real_scale=1))
        print(response)

        lf_imgs = (["/home/rnd/dev/opencv_training/test_images/lf/ref/sphere_1_ref0.png",
                   "/home/rnd/dev/opencv_training/test_images/lf/ref/sphere_1_ref1.png",
                   "/home/rnd/dev/opencv_training/test_images/lf/ref/sphere_1_ref2.png",
                   "/home/rnd/dev/opencv_training/test_images/lf/ref/sphere_1_ref3.png"])
        hf_imgs = (["/home/rnd/dev/opencv_training/test_images/hf/ref/sphere_20_ref0.png",
                   "/home/rnd/dev/opencv_training/test_images/hf/ref/sphere_20_ref1.png",
                   "/home/rnd/dev/opencv_training/test_images/hf/ref/sphere_20_ref2.png",
                   "/home/rnd/dev/opencv_training/test_images/hf/ref/sphere_20_ref3.png"])
        response = stub._ref_phase_compute(sla_pb2.tpu_env_req(lf_img = lf_imgs,
                                                               hf_img = hf_imgs))

        lf_imgs = (["/home/rnd/dev/opencv_training/test_images/lf/phase/sphere_1_phase0.png",
                   "/home/rnd/dev/opencv_training/test_images/lf/phase/sphere_1_phase1.png",
                   "/home/rnd/dev/opencv_training/test_images/lf/phase/sphere_1_phase2.png",
                   "/home/rnd/dev/opencv_training/test_images/lf/phase/sphere_1_phase3.png"])
        hf_imgs = (["/home/rnd/dev/opencv_training/test_images/hf/phase/sphere_20_phase0.png",
                   "/home/rnd/dev/opencv_training/test_images/hf/phase/sphere_20_phase1.png",
                   "/home/rnd/dev/opencv_training/test_images/hf/phase/sphere_20_phase2.png",
                   "/home/rnd/dev/opencv_training/test_images/hf/phase/sphere_20_phase3.png"])
        response = stub._depth_compute(sla_pb2.tpu_env_req(lf_img = lf_imgs,
                                                               hf_img = hf_imgs))

if __name__ == '__main__':
    logging.basicConfig()
    run()
