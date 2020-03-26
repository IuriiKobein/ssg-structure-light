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
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = sla_pb2_grpc.phase_unwrapStub(channel)
        response = stub.tpu_invoke(sla_pb2.tpu_req(url_ref_lf="ref1", url_ref_hf="ref2", url_obj_lf="obj1", url_obj_hf="obj2"))
    print("sla client received: " + response.url_unwrap_phase)


if __name__ == '__main__':
    logging.basicConfig()
    run()
