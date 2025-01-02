python -m grpc_tools.protoc -I ./ --proto_path=./clip.proto --python_out=. --grpc_python_out=./ clip.proto
