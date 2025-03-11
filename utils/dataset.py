import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor, AutoProcessor

from model.llama3_2 import conversation as conversation_lib
from model.llama3_2.constants import (IGNORE_INDEX, IMAGE_TOKEN_INDEX)
from model.llama3_2.mm_utils import prepare_messages_for_llama3_vision
from model.segment_anything.utils.transforms import ResizeLongestSide

from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json
from .reason_seg_dataset import ReasonSegDataset
from .refer import REFER
from .refer_seg_dataset import ReferSegDataset
from .sem_seg_dataset import SemSegDataset
from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN)
from .vqa_dataset import VQADataset


def collate_fn(
    batch, tokenizer=None, conv_type="llama_3", use_mm_start_end=True, local_rank=-1, processor=None
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        question,
        sampled_classes,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        if masks is not None:
            masks_list.append(masks.float())
        else:
            masks_list.append(None)
        if label is not None:
            label_list.append(label)
        else:
            label_list.append(None)
        resize_list.append(resize)
        if question is not None:
            questions_list.append(question)
        if sampled_classes is not None:
            sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    if tokenizer is not None:
        input_ids = []
        attention_masks = []
        labels = []

        if conv_type == "llama_3":
            # Llama3.2 vision用の処理
            for conv in conversation_list:
                # 空の入力テキストを準備
                empty_text = ""
                
                # 画像とテキストを組み合わせたメッセージ形式に変換
                messages = [{"role": "user", "content": [
                    {"type": "image"},  # 画像
                    {"type": "text", "text": empty_text}  # テキスト
                ]}]
                
                if processor is not None:
                    # プロセッサを使用してメッセージをエンコード
                    inputs = processor(text=empty_text, return_tensors="pt")
                    input_id = inputs["input_ids"][0]
                    attention_mask = inputs["attention_mask"][0]
                else:
                    # プロセッサがない場合はトークナイザで処理
                    input_id = tokenizer(empty_text).input_ids
                    input_id = torch.LongTensor(input_id)
                    attention_mask = torch.ones_like(input_id)
                
                # 入力とラベルを保存
                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                
                # ラベルは画像を除いた応答部分のみ
                label = torch.LongTensor([-100] * len(input_id))  # ラベルはIDENTITY_MASK（-100）でマスク
                labels.append(label)
        else:
            # 従来の処理方法（互換性のため残す）
            for conv in conversation_list:
                if isinstance(conv, list):
                    conv_text = conv
                else:
                    conv = conv.copy()
                    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

                    if use_mm_start_end:
                        if "llama3" in conv_type:
                            # Llama3形式での画像トークン処理
                            conv_text = []
                            for role, message in conv.messages:
                                if role == roles["human"]:
                                    if isinstance(message, list):
                                        # 画像とテキストを含むメッセージの場合
                                        message_text = ""
                                        for item in message:
                                            if isinstance(item, str):
                                                message_text += item
                                        conv_text.append(message_text)
                                    else:
                                        # テキストのみのメッセージの場合
                                        conv_text.append(message)
                                elif role == roles["gpt"]:
                                    conv_text.append(message)
                        else:
                            # 従来の形式での画像トークン処理
                            conv.messages = [
                                (
                                    role,
                                    message.replace(
                                        DEFAULT_IMAGE_TOKEN,
                                        f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}",
                                    ),
                                )
                                for role, message in conv.messages
                            ]
                            conv_text = conv.get_prompt()
                    else:
                        conv_text = conv.get_prompt()

                input_id = tokenizer_image_token(
                    conv_text, tokenizer, return_tensors="pt"
                )
                input_id = input_id.unsqueeze(0)
                attention_mask = torch.ones_like(input_id)
                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                # vitrually all the labels are -100, except for the response part from assistant
                # system instruction: -100
                # user input: -100
                # assistant response: actual token indices
                label = torch.ones_like(input_id) * IGNORE_INDEX
                if isinstance(conv, list):
                    pass
                else:
                    resp_token_ids = tokenizer_image_token(
                        conv.messages[-1][1], tokenizer, return_tensors="pt"
                    )
                    label[0, -len(resp_token_ids) :] = resp_token_ids
                labels.append(label)

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        # attention_masks = torch.ones_like(input_ids)
        labels = torch.cat(labels, dim=0)

        return {
            "image_paths": image_path_list,
            "images": images_list,
            "images_clip": images_clip_list,
            "input_ids": input_ids,
            "attention_masks": attention_masks,
            "labels": labels,
            "masks": masks_list,
            "label_list": label_list,
            "resize_list": resize_list,
            "offset": torch.LongTensor(offset_list),
            "questions": questions_list if len(questions_list) > 0 else None,
            "sampled_classes": sampled_classes_list
            if len(sampled_classes_list) > 0
            else None,
            "inference": inferences[0],
        }
    else:
        return {
            "image_paths": image_path_list,
            "images": images_list,
            "images_clip": images_clip_list,
            "masks": masks_list,
            "label_list": label_list,
            "resize_list": resize_list,
            "offset": torch.LongTensor(offset_list),
            "inference": inferences[0],
        }


class HybridDataset(torch.utils.data.Dataset):
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
        dataset="sem_seg||refer_seg||vqa||reason_seg",
        sample_rate=[9, 3, 3, 1],
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        vqa_data="llava_instruct_150k",
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
        processor=None,
    ):
        self.base_image_dir = base_image_dir
        self.tokenizer = tokenizer
        self.precision = precision
        self.image_size = image_size
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate, dtype="float64")
        self.sample_rate = sample_rate / sample_rate.sum()
        self.samples_per_epoch = samples_per_epoch
        if "llama" in vision_tower and "vision" in vision_tower:
            # Llama3.2 vision用のイメージプロセッサー
            self.transform = AutoProcessor.from_pretrained(vision_tower)
            self.is_llama3_vision = True
        else:
            # 従来のCLIPイメージプロセッサー
            self.transform = CLIPImageProcessor.from_pretrained(vision_tower)
            self.is_llama3_vision = False
            
        # プロセッサーの保持
        self.processor = processor
            
        # SAM用のリサイズ変換
        self.transform_sam = ResizeLongestSide(self.img_size)

        # 分割タイプとデータセットの初期化
        self.dataset_list = []
        dataset_names = dataset.split("||")
        print("Dataset:", dataset_names)
        
        assert len(dataset_names) == len(self.sample_rate)
        
        # セマンティックセグメンテーションデータセット
        if "sem_seg" in dataset_names:
            sem_seg_idx = dataset_names.index("sem_seg")
            if sem_seg_data is not None:
                self.sem_seg_dataset = SemSegDataset(
                    base_image_dir,
                    sem_seg_data,
                    tokenizer,
                    self.transform,
                    self.transform_sam,
                    num_classes_per_sample=num_classes_per_sample,
                    exclude_val=exclude_val,
                    processor=processor,
                )
                self.dataset_list.append(self.sem_seg_dataset)
            else:
                self.dataset_list.append(None)
                self.sample_rate[sem_seg_idx] = 0
        else:
            self.sem_seg_dataset = None

        # 参照セグメンテーションデータセット
        if "refer_seg" in dataset_names:
            refer_seg_idx = dataset_names.index("refer_seg")
            if refer_seg_data is not None:
                self.refer_seg_dataset = ReferSegDataset(
                    base_image_dir,
                    refer_seg_data,
                    tokenizer,
                    self.transform,
                    self.transform_sam,
                    processor=processor,
                )
                self.dataset_list.append(self.refer_seg_dataset)
            else:
                self.dataset_list.append(None)
                self.sample_rate[refer_seg_idx] = 0
        else:
            self.refer_seg_dataset = None

        # VQAデータセット
        if "vqa" in dataset_names:
            vqa_idx = dataset_names.index("vqa")
            if vqa_data is not None:
                self.vqa_dataset = VQADataset(
                    base_image_dir,
                    vqa_data,
                    tokenizer,
                    self.transform,
                    self.transform_sam,
                    processor=processor,
                )
                self.dataset_list.append(self.vqa_dataset)
            else:
                self.dataset_list.append(None)
                self.sample_rate[vqa_idx] = 0
        else:
            self.vqa_dataset = None

        # ReasonSegデータセット
        if "reason_seg" in dataset_names:
            reason_seg_idx = dataset_names.index("reason_seg")
            if reason_seg_data is not None:
                reason_seg_dataset_name, reason_seg_split = reason_seg_data.split("|")
                self.reason_seg_dataset = ReasonSegDataset(
                    base_image_dir,
                    reason_seg_dataset_name,
                    reason_seg_split,
                    tokenizer,
                    self.transform,
                    self.transform_sam,
                    explanatory,
                    processor=processor,
                )
                self.dataset_list.append(self.reason_seg_dataset)
            else:
                self.dataset_list.append(None)
                self.sample_rate[reason_seg_idx] = 0
        else:
            self.reason_seg_dataset = None

        # サンプリングレートの正規化
        if self.sample_rate.sum() == 0:
            self.sample_rate = np.ones_like(self.sample_rate) / len(self.sample_rate)
        else:
            self.sample_rate = self.sample_rate / self.sample_rate.sum()

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        dataset_idx = np.random.choice(len(self.dataset_list), p=self.sample_rate)
        # 選択されたデータセットからサンプルを取得
        image_path, image, image_clip, conversations, masks, label, resize, question, sampled_classes, inference = self.dataset_list[
            dataset_idx
        ].get_data()
        
        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            question,
            sampled_classes,
            inference,
        )


class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
        processor=None,
    ):
        self.base_image_dir = base_image_dir
        self.tokenizer = tokenizer
        self.image_size = image_size
        if "llama" in vision_tower and "vision" in vision_tower:
            # Llama3.2 vision用のイメージプロセッサー
            self.transform = AutoProcessor.from_pretrained(vision_tower)
            self.is_llama3_vision = True
        else:
            # 従来のCLIPイメージプロセッサー
            self.transform = CLIPImageProcessor.from_pretrained(vision_tower)
            self.is_llama3_vision = False
            
        # プロセッサーの保存
        self.processor = processor
        
        # SAM用のリサイズ変換
        self.transform_sam = ResizeLongestSide(self.img_size)

        self.val_dataset = val_dataset
        if self.val_dataset is not None:
            val_dataset_name, val_split = self.val_dataset.split("|")
            if val_dataset_name == "ReasonSeg":
                self.dataset = ReasonSegDataset(
                    base_image_dir,
                    val_dataset_name,
                    val_split,
                    tokenizer,
                    self.transform,
                    self.transform_sam,
                    0,
                    val=True,
                    processor=processor,
                )
            else:
                print(f"Val dataset {val_dataset_name} not supported!")
                self.dataset = None
                return

            self.ds_names = ["reasonseg"]
            # 検証用のデータファイルを読み込み
            self.all_annos = self.dataset.get_all_annos()
        else:
            self.dataset = None
            self.all_annos = []

    def __len__(self):
        return len(self.all_annos)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """SAM用の画像前処理"""
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        pad_size = max(h, w)
        padh = pad_size - h
        padw = pad_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        if self.dataset is not None:
            return self.dataset.get_anno_by_idx(idx, inference=True)
        return None
