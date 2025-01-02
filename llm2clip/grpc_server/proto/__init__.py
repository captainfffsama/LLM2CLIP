from packaging.version import Version
import os
import importlib

import grpc


class VersionPlus:
    def __init__(self, v: str):
        self._version: Version = Version(v)

    def __sub__(self, other):
        return (
            abs(self._version.major - other._version.major) * 100
            + abs(self._version.minor - other._version.minor)
            + abs(self._version.micro - other._version.micro) * 0.01
        )


_venv_grpc_version = grpc.__version__

_current_file_dir = os.path.dirname(__file__)
_close_grpc_version = None
_close_value = 1000
for item in os.listdir(_current_file_dir):
    item_path = os.path.join(_current_file_dir, item)
    if os.path.isdir(item_path) and item.startswith("v"):
        tmp_v = item.replace("_", ".")[1:]
        c = VersionPlus(tmp_v) - VersionPlus(_venv_grpc_version)
        if c < _close_value:
            _close_grpc_version = item
            _close_value = c

clip_pb2 = importlib.import_module(
    f"{__name__}.{_close_grpc_version}.clip_pb2"
)
clip_pb2_grpc = importlib.import_module(
    f"{__name__}.{_close_grpc_version}.clip_pb2_grpc"
)
print(f"venv grpc version is: {_venv_grpc_version}")
print(
    "will use grpc build code version is: {}".format(
        _close_grpc_version.replace("_", ".")
    )
)


# NOTE: 方便写代码用
#from .v1_66_1 import clip_pb2, clip_pb2_grpc  # noqa

__all__ = ["clip_pb2", "clip_pb2_grpc"]
