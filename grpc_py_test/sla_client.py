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

import argparse

def parse_args():
    sla_parser = argparse.ArgumentParser(description='frontend cli for SL service')
    commands = sla_parser.add_subparsers(title = "commands", dest="command")

    # SLA setup command
    setup = commands.add_parser('setup', help='create SL algorithms')
    setup.add_argument('--method', action='store', required=True, help='method category name')
    setup.add_argument('--width', action='store', type=int, required=True, help='image width')
    setup.add_argument('--height', action='store',type=int, required=True, help='image height')
    setup.add_argument('--freq_ratio', action='store', type=float, help='redundant')
    setup.add_argument('--real_scale', action='store', type=float, default=1.0, help='real scale')
    setup.add_argument('--num_of_periods', action='store', type=int, required=True, help='number of period in pattern')
    setup.add_argument('--use_markers', action='store_true', default=False, help='number of period in pattern')
    setup.add_argument('--horizontal_pattern', action='store_true', default=False, help='pattern orientation')
    setup.add_argument('--num_of_patterns', action='store', type=int, required=True, help='number of patterns')
    setup.add_argument('--num_of_pix_mark', action='store', default=70, type=int, help='number of patterns')
    setup.add_argument('--cv_method_id', action='store', type=int, help='OpenCV SL method id')

    compute_ref_phase = commands.add_parser('compute_ref_phase', help='compute reference phase')
    compute_ref_phase.add_argument('--method', action='store', required=True, help='method category name')
    compute_ref_phase.add_argument('--lf_image_list', action='append', help='method category name')
    compute_ref_phase.add_argument('--hf_image_list', action='append', required=True, help='method category name')

    compute_depth = commands.add_parser('compute_depth', help='compute object depth map')
    compute_depth.add_argument('--method', action='store', required=True, help='method category name')
    compute_depth.add_argument('--lf_image_list', action='append', help='method category name')
    compute_depth.add_argument('--hf_image_list', action='append', required=True, help='method category name')

    capture_compute_ref_phase = commands.add_parser('capture_compute_ref_phase', help='capture and compute reference phase')
    capture_compute_ref_phase.add_argument('--method', action='store', required=True, help='method category name')
    capture_compute_ref_phase.add_argument('--lf_image_list', action='append', help='method category name')
    capture_compute_ref_phase.add_argument('--hf_image_list', action='append', required=True, help='method category name')

    capture_compute_depth = commands.add_parser('capture_compute_depth', help='capture and compute object depth map')
    capture_compute_depth.add_argument('--method', action='store', required=True, help='method category name')
    capture_compute_depth.add_argument('--lf_image_list', action='append', help='method category name')
    capture_compute_depth.add_argument('--hf_image_list', action='append', required=True, help='method category name')

    return sla_parser.parse_args()

def create_opencv_conf_req(args):
    req = sla_pb2.conf_req()

    req.method = "sl_opencv"
    req.width = args.width
    req.height = args.height
    req.num_of_periods = args.num_of_periods
    req.use_markers = args.use_markers
    req.is_horizontal = args.horizontal_pattern
    req.num_of_patterns = 3
    req.num_of_pix_mark = args.num_of_pix_mark
    req.opencv_method_id = args.cv_method_id

    return req

def create_exper_conf_req(args):
    req = sla_pb2.conf_req()

    req.method = args.method
    req.width = args.width
    req.height = args.height
    req.num_of_periods = args.num_of_periods
    req.num_of_patterns = args.num_of_patterns
    req.freq_ratio = args.freq_ratio
    req.real_scale = args.real_scale

    return req

def create_conf_req(args):
    if args.method == 'sl_opencv':
        return create_opencv_conf_req(args)
    else:
        return create_exper_conf_req(args)

def setup_cmd_handler(sla_ctrl_strub, args):
    conf_req = create_conf_req(args)
    print(conf_req)
    response = sla_ctrl_strub._setup(conf_req)
    print(response.status)


def create_compute_req(args):
    req = sla_pb2.compute_req()

    req.method = args.method

    if args.lf_image_list:
        for img in args.lf_image_list:
            req.lf_img.append(img)

    for img in args.hf_image_list:
        req.hf_img.append(img)

    print(req)
    return req

def compute_ref_phase_cmd_handler(sla_ctrl_strub, args):
    conf_req = create_compute_req(args)

    response = sla_ctrl_strub._ref_phase_compute(conf_req)
    print(response)

def compute_depth_cmd_handler(sla_ctrl_strub, args):
    conf_req = create_compute_req(args)

    response = sla_ctrl_strub._depth_compute(conf_req)
    print(response)

def capture_compute_ref_phase_cmd_handler(sla_ctrl_strub, args):
    conf_req = create_compute_req(args)

    response = sla_ctrl_strub._ref_phase_capture_and_compute(conf_req)
    print(response)

def capture_compute_depth_cmd_handler(sla_ctrl_strub, args):
    conf_req = create_compute_req(args)

    response = sla_ctrl_strub._depth_capture_and_compute(conf_req)
    print(response)

def run(args):
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('localhost:12321') as channel:
        stub = sla_pb2_grpc.sla_ctrlStub(channel)
        if args.command == 'setup':
            setup_cmd_handler(stub, args)
        elif args.command == 'compute_ref_phase':
            compute_ref_phase_cmd_handler(stub, args)
        elif args.command == 'capture_compute_ref_phase':
            capture_compute_ref_phase_cmd_handler(stub, args)
        elif args.command == 'compute_depth':
            compute_depth_cmd_handler(stub, args)
        elif args.command == 'capture_compute_depth':
            capture_compute_depth_cmd_handler(stub, args)

if __name__ == '__main__':
    #logging.basicConfig()
    args = parse_args()
    run(args)
