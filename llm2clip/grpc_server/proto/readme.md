Automatically reference reliable gRPC compiled code according to the current environment. If there is no corresponding version in this folder, please ensure that the gRPC versions of the client and the server are consistent, and then regenerate the gRPC code.

# Steps for generating gRPC code
1. Determine the version to be generated, for example, 1.66.1.
2. Install the corresponding version, for example: `pip install grpcio==1.66.1 grpcio-tools==1.66.1`
3. Compile the code: `bash generate_proto_py.sh`
4. Create the corresponding folder and move the generated files into it. Add `__init__.py`, and pay attention to naming the folder as per the example:
```bash
mkdir v1_66_1
mv clip_pb2*.py ./v1_66_1
cd v1_66_1
touch __init__.py
```
5. Change `from . import clip_pb2 as clip__pb2` in `clip_pb2_grpc.py` to `from . import clip_pb2 as clip__pb2`.