import json
import os
import random
import glob
import numpy as np

import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from model.llama3_2 import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import DEFAULT_IMAGE_TOKEN


def preprocess_multimodal(source, mm_use_im_start_end):
    for sentence in source:
        if DEFAULT_IMAGE_TOKEN in sentence["value"]:
            sentence["value"] = (
                sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
            )
            sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
            sentence["value"] = sentence["value"].strip()
            if "mmtag" in conversation_lib.default_conversation.version:
                sentence["value"] = sentence["value"].replace(
                    DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>"
                )
    return source


class VQADataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        vqa_data="llava_instruct_150k",
        processor=None,
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        
        self.processor = processor
        if self.processor is None:
            raise ValueError("processorが指定されていません。Llama3.2 VisionモデルではAutoProcessorが必須です。");

        DATA_DIR = os.path.join(base_image_dir, "llava_dataset")
        self.vqa_image_root = os.path.join(base_image_dir, "coco/train2017")
        
        # JSON データをロード
        with open(os.path.join(DATA_DIR, "{}.json".format(vqa_data))) as f:
            vqa_data_full = json.load(f)
        
        # 実際に存在する画像ファイルの一覧を取得
        available_images = []
        if os.path.exists(self.vqa_image_root):
            available_images = [os.path.basename(f) for f in glob.glob(os.path.join(self.vqa_image_root, "*.jpg"))]
            print(f"Found {len(available_images)} available images in {self.vqa_image_root}")
        else:
            print(f"WARNING: VQA image directory not found: {self.vqa_image_root}")
        
        # 存在する画像ファイルだけをフィルタリング
        self.vqa_data = []
        if available_images:
            for item in vqa_data_full:
                if item["image"] in available_images:
                    self.vqa_data.append(item)
            
            print(f"Filtered VQA data from {len(vqa_data_full)} to {len(self.vqa_data)} items based on available images")
        else:
            # 画像が見つからない場合は空のリストを使用
            print("WARNING: No available images found for VQA dataset")
            self.vqa_data = []

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        # データセットが空の場合は明示的なエラーを返す
        if len(self.vqa_data) == 0:
            raise ValueError("VQAデータセットが空です。small_test_datasetにはLLaVAデータに対応するCOCO画像が含まれていません。")
            
        idx = random.randint(0, len(self.vqa_data) - 1)
        item = self.vqa_data[idx]
        image_path = os.path.join(self.vqa_image_root, item["image"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        
        # 画像のプリプロセス
        image_pil = Image.fromarray(image)
        processed = self.processor(images=image_pil, return_tensors="pt")
        image_clip = processed.pixel_values[0]

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        conv = conversation_lib.default_conversation.copy()
        source = item["conversations"]
        source = preprocess_multimodal(
            source,
            mm_use_im_start_end=conv.sep_style == conversation_lib.SeparatorStyle.TWO,
        )
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conversations = []
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

        questions = conversations
        sampled_classes = conversations

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        masks = torch.rand(0, *ori_size)
        label = torch.ones(ori_size) * self.ignore_label

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
        )
