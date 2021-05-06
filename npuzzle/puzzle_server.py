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
"""The Python implementation of the GRPC helloworld.Greeter server."""

from concurrent import futures
import logging

import grpc

import puzzle_worker_pb2
import puzzle_worker_pb2_grpc


class Puzzler(puzzle_worker_pb2_grpc.PuzzlerServicer):

    def FindPuzzle(self, request, context):
        print(f'Received puzzle message {request.puzzle}!')
        return puzzle_worker_pb2.PuzzleReply(path='Hello, %s!' % request.puzzle)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    puzzle_worker_pb2_grpc.add_PuzzlerServicer_to_server(Puzzler(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()