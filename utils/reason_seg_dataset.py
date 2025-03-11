import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model.llama3_2 import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
                    EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
                    SHORT_QUESTION_LIST)


class ReasonSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=12800,
        precision="fp32",
        image_size=224,
        num_classes_per_sample=1,
        exclude_val=False,
        reason_seg_data=None,
        explanatory=0.0,
        processor=None,
    ):
        """
        ReasonSegDatasetの初期化
        Args:
            base_image_dir: ベースとなる画像ディレクトリ
            tokenizer: テキストのトークナイザー
            vision_tower: ビジョンモデル
            samples_per_epoch: エポックあたりのサンプル数
            precision: 精度
            image_size: 画像サイズ
            num_classes_per_sample: サンプルあたりのクラス数
            exclude_val: 検証データを除外するかどうか
            reason_seg_data: ReasonSegデータへのパス（ディレクトリまたはデータリスト）
            explanatory: 説明付きデータの割合
            processor: Llama3.2 Visionモデル用のプロセッサー
        """
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.explanatory = explanatory
        self.img_size = 1024  # SAM用の画像サイズ

        # Llama3.2 Vision用のプロセッサーが必要
        if processor is None:
            raise ValueError("AutoProcessor is required for Llama3.2 Vision")
        self.processor = processor

        # 画像前処理のパラメータ
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.img_size = 1024 
        self.ignore_label = 255

        # SAM用の画像変換
        self.transform = ResizeLongestSide(self.img_size)

        # 質問と回答のテンプレート
        self.short_question_list = [
            "What is {}?",
            "Can you segment {}?",
            "Please segment {}.",
            "Segment {}.",
            "Show me {}.",
        ]
        self.long_question_list = [
            "What is {}?",
            "Could you segment what I'm referring to when I say {}?",
            "Please segment what I mean by {}.",
            "Segment what I mean by {}.",
            "Segment the region I'm referring to when I say {}.",
            "Can you segment what I mean by {}?",
        ]
        self.explanatory_question_list = [
            "Could you explain why?",
            "Why is that?",
            "Could you explain your reasoning?",
            "How did you determine that?",
        ]
        self.answer_list = [
            "Sure, it is <seg>.",
            "Yes, it is <seg>.",
            "It is <seg>.",
            "The region is <seg>.",
            "<seg>",
        ]

        # データセットの初期化
        self.image_paths = []
        self.json_paths = []
        self.explanatory_image_paths = []
        self.explanatory_json_paths = []

        # reason_seg_dataの処理
        if reason_seg_data is not None:
            # ディレクトリが指定された場合
            if isinstance(reason_seg_data, str):
                reason_seg_dir = os.path.join(base_image_dir, reason_seg_data)
                if not os.path.exists(reason_seg_dir):
                    print(f"WARNING: ReasonSeg directory not found: {reason_seg_dir}")
                    # 他の可能性のあるパスを試す
                    alt_path = os.path.join(base_image_dir, "reason_seg/ReasonSeg")
                    if os.path.exists(alt_path):
                        reason_seg_dir = alt_path
                        print(f"Using alternative path for ReasonSeg: {alt_path}")
                    else:
                        print(f"WARNING: No valid ReasonSeg directory found")
                        return

                # 訓練データセットを収集
                train_dir = os.path.join(reason_seg_dir, "train")
                if os.path.exists(train_dir):
                    # 画像とJSONのペアを収集
                    for file in os.listdir(train_dir):
                        if file.endswith(".jpg") or file.endswith(".png"):
                            img_path = os.path.join(train_dir, file)
                            json_file = file.rsplit(".", 1)[0] + ".json"
                            json_path = os.path.join(train_dir, json_file)
                            if os.path.exists(json_path):
                                self.image_paths.append(img_path)
                                self.json_paths.append(json_path)
                
                # 検証データも含める場合
                if not exclude_val:
                    val_dir = os.path.join(reason_seg_dir, "val")
                    if os.path.exists(val_dir):
                        for file in os.listdir(val_dir):
                            if file.endswith(".jpg") or file.endswith(".png"):
                                img_path = os.path.join(val_dir, file)
                                json_file = file.rsplit(".", 1)[0] + ".json"
                                json_path = os.path.join(val_dir, json_file)
                                if os.path.exists(json_path):
                                    self.image_paths.append(img_path)
                                    self.json_paths.append(json_path)

                # explanatory画像の収集
                if explanatory > 0:
                    explanatory_dir = os.path.join(reason_seg_dir, "explanatory")
                    if os.path.exists(explanatory_dir):
                        for file in os.listdir(explanatory_dir):
                            if file.endswith(".jpg") or file.endswith(".png"):
                                img_path = os.path.join(explanatory_dir, file)
                                json_file = file.rsplit(".", 1)[0] + ".json"
                                json_path = os.path.join(explanatory_dir, json_file)
                                if os.path.exists(json_path):
                                    self.explanatory_image_paths.append(img_path)
                                    self.explanatory_json_paths.append(json_path)
            # データリストが直接渡された場合
            elif isinstance(reason_seg_data, tuple) and len(reason_seg_data) == 2:
                images, jsons = reason_seg_data
                if len(images) != len(jsons):
                    print(f"WARNING: Number of images ({len(images)}) != number of JSONs ({len(jsons)})")
                for img_path, json_path in zip(images, jsons):
                    if os.path.exists(img_path) and os.path.exists(json_path):
                        self.image_paths.append(img_path)
                        self.json_paths.append(json_path)
        
        print(f"Initialized ReasonSegDataset with {len(self.image_paths)} samples and {len(self.explanatory_image_paths)} explanatory samples")

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
        # データセットが空でない場合のみ処理
        if not self.image_paths:
            # ダミーデータを返す
            dummy_image = torch.zeros(3, self.image_size, self.image_size)
            dummy_mask = torch.zeros(1, self.img_size, self.img_size)
            return {
                "image": dummy_image,
                "image_clip": dummy_image,
                "masks": [dummy_mask],
                "conversations": [],
                "class_names": ["dummy"],
                "resize": (self.image_size, self.image_size),
                "original_size": (self.img_size, self.img_size),
            }

        # ランダムに画像-JSONペアを選択
        idx = random.randint(0, len(self.image_paths) - 1)
        image_path = self.image_paths[idx]
        json_path = self.json_paths[idx]

        # 画像読み込み
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"WARNING: Failed to load image: {image_path}")
                # 別の方法で画像を読み込む
                image = np.array(Image.open(image_path))
                if len(image.shape) == 2:  # グレースケール画像の場合
                    image = np.stack([image, image, image], axis=2)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"ERROR: Failed to load image {image_path}: {e}")
            # ダミー画像を作成
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        # オリジナルサイズを保存
        ori_size = image.shape[:2]
        
        # 画像のプリプロセス
        image_pil = Image.fromarray(image)
        processed = self.processor(images=image_pil, return_tensors="pt")
        image_clip = processed.pixel_values[0]

        # マスクとセンテンスの取得
        try:
            mask, sents, is_sentence = get_mask_from_json(json_path, image)
        except Exception as e:
            print(f"ERROR: Failed to load mask from JSON {json_path}: {e}")
            # ダミーデータを作成
            mask = np.zeros(ori_size, dtype=np.uint8)
            sents = ["Error loading data"]
            is_sentence = True

        # サンプリング
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
        
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        sampled_masks = [
            (mask == 1).astype(np.float32) for _ in range(len(sampled_inds))
        ]

        # SAM用に画像を変換
        image = self.transform.apply_image(image)
        resize = image.shape[:2]

        # explanatory（説明）モードの処理
        use_explanatory = False
        if self.explanatory_image_paths and random.random() < self.explanatory:
            use_explanatory = True
            
            # ランダムに説明用画像-JSONペアを選択
            exp_idx = random.randint(0, len(self.explanatory_image_paths) - 1)
            exp_image_path = self.explanatory_image_paths[exp_idx]
            exp_json_path = self.explanatory_json_paths[exp_idx]
            
            # ここで説明用の画像とJSONを処理する
            # 必要に応じて追加の処理を実装
            
            # 質問タイプの選択 (explanatory)
            if is_sentence:
                question = self.explanatory_question_list[
                    random.randint(0, len(self.explanatory_question_list) - 1)
                ]
            else:
                return self.__getitem__(idx)
        else:
            # 通常の質問タイプの選択
            if is_sentence:
                if len(sampled_sents[0].split()) > 5:
                    question = self.long_question_list[
                        random.randint(0, len(self.long_question_list) - 1)
                    ]
                else:
                    question = self.short_question_list[
                        random.randint(0, len(self.short_question_list) - 1)
                    ]
            else:
                question = "What is"

        # 会話形式のデータを作成
        conversations = []
        conv = conversation_lib.default_conversation.copy()
        
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        
        # 人間の質問
        human_query = question + " " + sampled_sents[0]
        conversations.append({
            "from": roles["human"],
            "value": f"{human_query}"
        })
        
        # GPTの回答（<SEG>トークンを含む）
        gpt_answer = self.answer_list[random.randint(0, len(self.answer_list) - 1)]
        conversations.append({
            "from": roles["gpt"],
            "value": f"{gpt_answer}"
        })

        # 最終的なデータを返す
        return {
            "image": torch.from_numpy(image).permute(2, 0, 1).float(),
            "image_clip": image_clip,
            "masks": [torch.from_numpy(mask).float() for mask in sampled_masks],
            "conversations": conversations,
            "class_names": sampled_sents,
            "resize": resize,
            "original_size": ori_size,
        }
