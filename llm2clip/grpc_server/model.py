import datetime
from transformers import AutoModel, AutoConfig, AutoTokenizer
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from eva_clip import create_model_and_transforms
from llm2vec import LLM2Vec
import torch
from grpc_server.proto import clip_pb2_grpc
from grpc_server.utils import load_image_from_base64, np2tensor_proto

MODEL_ARGS_MAP = {"EVA02-CLIP-L-14-336": {}}


class LLM2Clip(clip_pb2_grpc.CLIPServiceServicer):
    def __init__(
        self,
        clip_name: str,
        ckpt_path: str,
        llm_model_name: str,
        l2v_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    ):
        self.model, _, self.preprocess = create_model_and_transforms(
            clip_name,
            force_custom_clip=True,
        )
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt)
        self.model = self.model.cuda().eval()

        config = AutoConfig.from_pretrained(llm_model_name, trust_remote_code=True)
        llm_model = AutoModel.from_pretrained(
            llm_model_name,
            torch_dtype=torch.bfloat16,
            config=config,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        llm_model.config._name_or_path = l2v_name  #  Workaround for LLM2VEC
        self.l2v = (
            LLM2Vec(
                llm_model,
                tokenizer,
                pooling_mode="mean",
                max_length=512,
                doc_max_length=512,
            )
            .cuda()
            .eval()
        )

    def image_em_get(self, img):
        image = self.preprocess(img).cuda().unsqueeze(0)
        image_features: torch.Tensor = self.model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def text_em_get(self, captions):
        text_features = self.l2v.encode(captions, convert_to_tensor=True).cuda()
        text_features = self.model.encode_text(text_features)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def ImageEmbeddingGet(self, request, context):
        image = load_image_from_base64(request.imdata)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.image_em_get(image)
            image_features = image_features.detach().cpu().numpy()
        print(
            "{} image embedding size: ".format(datetime.datetime.now()),
            image_features.shape,
        )
        torch.cuda.empty_cache()
        return np2tensor_proto(image_features)

    def TextEmbeddingGet(self, request, context):
        captions = [x for x in request.textdata]
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.text_em_get(captions)
            text_features = text_features.detach().cpu().numpy()
        print(
            "{} text embedding size: ".format(datetime.datetime.now()),
            text_features.shape,
        )
        torch.cuda.empty_cache()
        return np2tensor_proto(text_features)

    def ZeroShotCls(self, request, context):
        image = load_image_from_base64(request.imdata.imdata)
        captions = [x for x in request.textdata.textdata]
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.image_em_get(image)
            text_features = self.text_em_get(captions)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            text_probs = text_probs.detach().cpu().numpy()
        print("{} zero shor cls softmatx: ".format(datetime.datetime.now()), text_probs)
        return np2tensor_proto(text_probs)
