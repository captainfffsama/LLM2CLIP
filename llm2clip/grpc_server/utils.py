from typing import Union
import base64
import io

import numpy as np
from PIL import Image

from proto import clip_pb2

def img2base64(file: Union[str, np.ndarray]) -> bytes:
    if isinstance(file, str):
        img_file = open(file, "rb")  # 二进制打开图片文件
        img_b64encode = base64.b64encode(img_file.read())  # base64编码
        img_file.close()  # 文件关闭
        return img_b64encode
    elif isinstance(file, np.ndarray):
        img = Image.fromarray(file)
        buffered = io.BytesIO()
        img.save(buffered, format=".jpg")
        img_b64encode = base64.b64encode(buffered.getvalue())
        buffered.close()
        return img_b64encode
    else:
        return None
def load_image_from_base64(base64_string):
    image_data = base64.b64decode(base64_string)
    image_file = io.BytesIO(image_data)
    image = Image.open(image_file)
    return image


def np2tensor_proto(np_ndarray: np.ndarray):
    shape = list(np_ndarray.shape)
    data = np_ndarray.flatten().tolist()
    tensor_pb =clip_pb2.Tensor()
    tensor_pb.shape.extend(shape)
    tensor_pb.data.extend(data)
    return tensor_pb


def tensor_proto2np(tensor_pb):
    np_matrix = np.array(tensor_pb.data, dtype=float).reshape(tensor_pb.shape)
    return np_matrix
