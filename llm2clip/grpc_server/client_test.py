import grpc
from proto import clip_pb2_grpc, clip_pb2
from utils import img2base64, tensor_proto2np
import numpy as np

channel_opt = [
    ("grpc.max_send_message_length", 512 * 1024 * 1024),
    ("grpc.max_receive_message_length", 512 * 1024 * 1024),
]
channel = grpc.insecure_channel("127.0.0.1:52009", options=channel_opt)
stub = clip_pb2_grpc.CLIPServiceStub(channel)

image_path = "/home/chiebot-cv/disk/disk1/hq_workspace/pl_jyz_vlm_test/game_a8fe1e3f7db09e45915c5f205e18494a_jyz_0.jpg"
# captions = ["a diagram", "a dog", "a cat","a in"]
captions = [
    "a diagram",
    "a dog",
    "a cat",
    # "a broken electric insulator",
    "a intact insulators",
    "a damaged insulators",
    "a rings",
]

image_req = clip_pb2.ImageEmbeddingRequest()
image_req.imdata=img2base64(image_path)
txt_req = clip_pb2.TextEmbeddingRequest()
for cap in captions:
    txt_req.textdata.append(cap)

all_req = clip_pb2.ImageTextData(imdata=image_req, textdata=txt_req)


def softmax(x, dim=None):
    x = np.asarray(x)
    if not (dim is None or isinstance(dim, int)):
        raise ValueError("Dim must be none or an integer.")
    if dim is None:
        x = x - np.max(x)  # 数值稳定
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)
    if dim < -x.ndim or dim >= x.ndim:
        raise ValueError(f"Dim {dim} out of range for array with shape {x.shape}.")
    if dim < 0:
        dim = x.ndim + dim
    x_max = np.max(x, axis=dim, keepdims=True)
    x = x - x_max  # 数值稳定
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x, axis=dim, keepdims=True)
    return exp_x / sum_exp_x


img_em = stub.ImageEmbeddingGet(image_req)
txt_em = stub.TextEmbeddingGet(txt_req)
img_em = tensor_proto2np(img_em)
txt_em = tensor_proto2np(txt_em)
txt_prob = softmax(100 * img_em @ txt_em.T,dim=-1)
print("txt_prob:",txt_prob)

txt_prob=stub.ZeroShotCls(all_req)
txt_prob=tensor_proto2np(txt_prob)

print("txt_prob:",txt_prob)
