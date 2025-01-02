import os
import argparse
from concurrent import futures
import asyncio

import grpc
from proto import clip_pb2_grpc

import base_config as config_manager


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c", "--cfg", type=str, default="")
    args = parser.parse_args()
    return args


async def main(grpc_args, model):
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=grpc_args["max_workers"]),
        options=[
            ("grpc.max_send_message_length", grpc_args["max_send_message_length"]),
            (
                "grpc.max_receive_message_length",
                grpc_args["max_receive_message_length"],
            ),
        ],
    )
    clip_pb2_grpc.add_CLIPServiceServicer_to_server(model, server)

    server.add_insecure_port("{}:{}".format(grpc_args["host"], grpc_args["port"]))
    await server.start()
    print("llm2clip gprc server init done")
    await server.wait_for_termination()


def run():
    args = parse_args()
    if os.path.exists(args.cfg):
        config_manager.merge_param(args.cfg)
    args_dict: dict = config_manager.param
    detector_params = args_dict["model_params"]
    if detector_params["device"].startswith("cuda"):
        if detector_params["device"] == "cuda":
            device_num = 0
        else:
            device_num = detector_params["device"].split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = device_num
    detector_params.pop("device")

    from model import LLM2Clip

    model = LLM2Clip(**detector_params)
    print("model init done!")
    grpc_args = args_dict["grpc_args"]
    asyncio.run(main(grpc_args, model))


if __name__ == "__main__":
    run()
