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
    for batch_idx, (
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
    ) in enumerate(batch):
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        if masks is not None:
            masks_list.append(masks.float())
        else:
            # マスクがNoneの場合、詳細なデバッグ情報をログに表示
            print(f"[データ解析] バッチインデックス {batch_idx} でマスクがNoneです。")
            print(f"[データ解析]   - 画像パス: {image_path}")
            print(f"[データ解析]   - 会話数: {len(conversations)}")
            print(f"[データ解析]   - リサイズ値: {resize}")
            if question is not None:
                print(f"[データ解析]   - 質問: {question}")
            if sampled_classes is not None:
                print(f"[データ解析]   - サンプルクラス: {sampled_classes}")
            masks_list.append(None)
        if label is not None:
            label_list.append(label)
        else:
            # ラベルがNoneの場合、詳細なデバッグ情報をログに表示
            print(f"[データ解析] バッチインデックス {batch_idx} でラベルがNoneです。")
            print(f"[データ解析]   - 画像パス: {image_path}")
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
                if isinstance(conv, list):
                    # テキストリストの場合はテキスト部分を抽出
                    text_content = conv[0] if len(conv) > 0 else ""
                else:
                    # 会話オブジェクトの場合はプロンプトを取得
                    text_content = conv.get_prompt() if hasattr(conv, "get_prompt") else ""
                
                # processorが提供されている場合はprocessorを使用
                if processor is not None:
                    # プロセッサを使用してテキストをエンコード（画像は後でバッチ処理で追加）
                    chat_template = [{"role": "user", "content": text_content}]
                    chat_text = processor.tokenizer.apply_chat_template(
                        chat_template, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    
                    inputs = processor.tokenizer(
                        chat_text, 
                        return_tensors="pt", 
                        padding="longest",
                        truncation=True
                    )
                    
                    input_id = inputs["input_ids"][0]
                    attention_mask = inputs["attention_mask"][0]
                    
                    # ラベルを作成 - 入力部分は-100でマスク
                    label = torch.ones_like(input_id) * -100
                    
                    # <SEG>トークンが応答に含まれている場合は、そのIDをラベルに設定
                    if "<SEG>" in text_content and tokenizer is not None:
                        seg_token_id = tokenizer.convert_tokens_to_ids("<SEG>")
                        if seg_token_id is not None:
                            # <SEG>トークンを含む応答部分を抽出
                            seg_positions = (input_id == seg_token_id).nonzero(as_tuple=True)[0]
                            for pos in seg_positions:
                                label[pos] = seg_token_id
                else:
                    # プロセッサがない場合はトークナイザで処理（フォールバック）
                    input_id = tokenizer(text_content, return_tensors="pt").input_ids[0]
                    attention_mask = torch.ones_like(input_id)
                    label = torch.ones_like(input_id) * -100
                
                # 入力とラベルを保存
                input_ids.append(input_id)
                attention_masks.append(attention_mask)
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

        # 画像をスタックする（すべての画像が同じ形状の場合のみ）
        if all(x.shape == images_list[0].shape for x in images_list):
            stacked_images = torch.stack(images_list, dim=0)
        else:
            print("WARNING: Images have different shapes, cannot stack them")
            stacked_images = images_list
            
        if all(x.shape == images_clip_list[0].shape for x in images_clip_list):
            stacked_images_clip = torch.stack(images_clip_list, dim=0)
        else:
            print("WARNING: CLIP images have different shapes, cannot stack them")
            stacked_images_clip = images_clip_list

        return {
            "image_paths": image_path_list,
            "images": stacked_images,
            "images_clip": stacked_images_clip,
            "input_ids": input_ids,
            "attention_masks": attention_masks,
            "labels": labels,
            "masks_list": masks_list,
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
        # 画像をスタックする（すべての画像が同じ形状の場合のみ）
        if all(x.shape == images_list[0].shape for x in images_list):
            stacked_images = torch.stack(images_list, dim=0)
        else:
            print("WARNING: Images have different shapes, cannot stack them")
            stacked_images = images_list
            
        if all(x.shape == images_clip_list[0].shape for x in images_clip_list):
            stacked_images_clip = torch.stack(images_clip_list, dim=0)
        else:
            print("WARNING: CLIP images have different shapes, cannot stack them")
            stacked_images_clip = images_clip_list
            
        return {
            "image_paths": image_path_list,
            "images": stacked_images,
            "images_clip": stacked_images_clip,
            "masks_list": masks_list,
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
                try:
                    print(f"Initializing SemSegDataset with base_dir={base_image_dir}, data={sem_seg_data}")
                    self.sem_seg_dataset = SemSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch=samples_per_epoch // 4,
                        precision=precision,
                        image_size=image_size,
                        num_classes_per_sample=num_classes_per_sample,
                        exclude_val=exclude_val,
                        sem_seg_data=sem_seg_data,
                        processor=processor,
                    )
                    self.dataset_list.append(self.sem_seg_dataset)
                except Exception as e:
                    print(f"ERROR: Failed to initialize SemSegDataset: {e}")
                    import traceback
                    traceback.print_exc()
                    print("Creating dummy dataset for sem_seg")
                    self.sem_seg_dataset = None
                    self.dataset_list.append(None)
                    self.sample_rate[sem_seg_idx] = 0
            else:
                self.dataset_list.append(None)
                self.sample_rate[sem_seg_idx] = 0
        else:
            self.sem_seg_dataset = None

        # 参照セグメンテーションデータセット
        if "refer_seg" in dataset_names:
            refer_seg_idx = dataset_names.index("refer_seg")
            if refer_seg_data is not None:
                try:
                    print(f"Initializing ReferSegDataset with base_dir={base_image_dir}, data={refer_seg_data}")
                    self.refer_seg_dataset = ReferSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch=samples_per_epoch // 4,
                        precision=precision,
                        image_size=image_size,
                        num_classes_per_sample=num_classes_per_sample, 
                        exclude_val=exclude_val,
                        refer_seg_data=refer_seg_data,
                        processor=processor,
                    )
                    self.dataset_list.append(self.refer_seg_dataset)
                except Exception as e:
                    print(f"ERROR: Failed to initialize ReferSegDataset: {e}")
                    import traceback
                    traceback.print_exc()
                    print("Creating dummy dataset for refer_seg")
                    self.refer_seg_dataset = None
                    self.dataset_list.append(None)
                    self.sample_rate[refer_seg_idx] = 0
            else:
                self.dataset_list.append(None)
                self.sample_rate[refer_seg_idx] = 0
        else:
            self.refer_seg_dataset = None

        # VQAデータセット
        if "vqa" in dataset_names:
            vqa_idx = dataset_names.index("vqa")
            if vqa_data is not None:
                try:
                    print(f"Initializing VQADataset with base_dir={base_image_dir}, data={vqa_data}")
                    self.vqa_dataset = VQADataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch=samples_per_epoch // 4,
                        precision=precision,
                        image_size=image_size,
                        num_classes_per_sample=num_classes_per_sample,
                        exclude_val=exclude_val,
                        vqa_data=vqa_data,
                        processor=processor,
                    )
                    self.dataset_list.append(self.vqa_dataset)
                except Exception as e:
                    print(f"ERROR: Failed to initialize VQADataset: {e}")
                    import traceback
                    traceback.print_exc()
                    print("Creating dummy dataset for vqa")
                    self.vqa_dataset = None
                    self.dataset_list.append(None)
                    self.sample_rate[vqa_idx] = 0
            else:
                self.dataset_list.append(None)
                self.sample_rate[vqa_idx] = 0
        else:
            self.vqa_dataset = None

        # ReasonSegデータセット
        if "reason_seg" in dataset_names:
            reason_seg_idx = dataset_names.index("reason_seg")
            try:
                print(f"Initializing ReasonSegDataset with base_dir={base_image_dir}")
                self.reason_seg_dataset = ReasonSegDataset(
                    base_image_dir,
                    tokenizer,
                    vision_tower,
                    samples_per_epoch=samples_per_epoch // 4,
                    precision=precision,
                    image_size=image_size,
                    num_classes_per_sample=num_classes_per_sample,
                    exclude_val=exclude_val,
                    reason_seg_data="reason_seg/ReasonSeg",
                    explanatory=0.1,
                    processor=processor,
                )
                self.dataset_list.append(self.reason_seg_dataset)
            except Exception as e:
                print(f"ERROR: Failed to initialize ReasonSegDataset: {e}")
                import traceback
                traceback.print_exc()
                print("Creating dummy dataset for reason_seg")
                self.reason_seg_dataset = None
                self.dataset_list.append(None)
                self.sample_rate[reason_seg_idx] = 0
        else:
            self.reason_seg_dataset = None

        # 有効なデータセットがあるか確認
        valid_datasets = [ds for ds in self.dataset_list if ds is not None]
        if not valid_datasets:
            print("WARNING: No valid datasets were initialized!")
            print(f"Dataset names: {dataset_names}")
            print(f"Sample rates: {sample_rate}")
            # 少なくとも1つのダミーデータセットを作成（エラーを回避するため）
            print("Creating a dummy dataset to avoid runtime errors")
            # 最初のデータセット名を取得
            if dataset_names:
                first_ds = dataset_names[0]
                print(f"Creating dummy dataset for {first_ds}")
                # 必要なダミーデータセットを作成（実装が必要）
        
        # サンプリングレートの正規化
        if self.sample_rate.sum() == 0:
            self.sample_rate = np.ones_like(self.sample_rate) / len(self.sample_rate)
        else:
            self.sample_rate = self.sample_rate / self.sample_rate.sum()

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # 再帰呼び出しの回数を制限するためのカウンタを追加
        call_depth = getattr(self, '_call_depth', 0)
        setattr(self, '_call_depth', call_depth + 1)
        
        # 再帰呼び出しが多すぎる場合、ダミーデータを返す
        if call_depth > 10:
            print("警告: HybridDatasetの__getitem__メソッドでの再帰呼び出しが多すぎます。ダミーデータを返します。")
            setattr(self, '_call_depth', 0)  # カウンタをリセット
            dummy_image = torch.zeros(3, 224, 224)
            dummy_image_clip = torch.zeros(3, 224, 224)
            dummy_masks = torch.zeros(1, 224, 224)
            dummy_label = torch.ones(224, 224) * self.ignore_label
            dummy_conversations = ["ダミーデータ"]
            return (
                "dummy_path",
                dummy_image,
                dummy_image_clip,
                dummy_conversations,
                dummy_masks,
                dummy_label,
                (224, 224),
                None,
                None,
                False,
            )
            
        dataset_idx = np.random.choice(len(self.dataset_list), p=self.sample_rate)
        
        # 選択されたデータセットがNoneの場合は別のデータセットを試みる
        max_attempts = 5
        attempts = 0
        
        while self.dataset_list[dataset_idx] is None and attempts < max_attempts:
            print(f"警告: 選択されたデータセット {dataset_idx} は None です。別のデータセットを試みます。")
            # サンプリングレートを一時的に更新（選択されたデータセットを除外）
            temp_rates = self.sample_rate.copy()
            temp_rates[dataset_idx] = 0
            if np.sum(temp_rates) > 0:
                temp_rates = temp_rates / np.sum(temp_rates)
                dataset_idx = np.random.choice(len(self.dataset_list), p=temp_rates)
            else:
                # すべてのデータセットがNoneの場合
                print("エラー: すべての有効なデータセットがNoneです。ダミーデータを返します。")
                setattr(self, '_call_depth', 0)  # カウンタをリセット
                dummy_image = torch.zeros(3, 224, 224)
                dummy_image_clip = torch.zeros(3, 224, 224)
                dummy_masks = torch.zeros(1, 224, 224)
                dummy_label = torch.ones(224, 224) * self.ignore_label
                dummy_conversations = ["すべてのデータセットが無効です"]
                return (
                    "dummy_path",
                    dummy_image,
                    dummy_image_clip,
                    dummy_conversations,
                    dummy_masks,
                    dummy_label,
                    (224, 224),
                    None,
                    None,
                    False,
                )
            attempts += 1
            
        # すべての試行後もデータセットがNoneの場合
        if self.dataset_list[dataset_idx] is None:
            print("エラー: 有効なデータセットが見つかりませんでした。ダミーデータを返します。")
            setattr(self, '_call_depth', 0)  # カウンタをリセット
            dummy_image = torch.zeros(3, 224, 224)
            dummy_image_clip = torch.zeros(3, 224, 224)
            dummy_masks = torch.zeros(1, 224, 224)
            dummy_label = torch.ones(224, 224) * self.ignore_label
            dummy_conversations = ["有効なデータセットが見つかりません"]
            return (
                "dummy_path",
                dummy_image,
                dummy_image_clip,
                dummy_conversations,
                dummy_masks,
                dummy_label,
                (224, 224),
                None,
                None,
                False,
            )
        
        try:
            # 選択されたデータセットからサンプルを取得
            max_data_attempts = 3
            data_attempts = 0
            data = None
            
            while data is None and data_attempts < max_data_attempts:
                try:
                    data = self.dataset_list[dataset_idx][random.randint(0, 10)]  # データセットからランダムにサンプルを取得
                    data_attempts += 1
                except Exception as e:
                    print(f"エラー: データセット[{dataset_idx}]からのサンプル取得に失敗しました（試行 {data_attempts+1}/{max_data_attempts}）: {e}")
                    if data_attempts >= max_data_attempts - 1:
                        raise  # 最後の試行でも失敗した場合は例外を再発生
            
            # データの要素数をチェック
            if isinstance(data, tuple) and len(data) == 9:
                # 9要素のタプルの場合、inferenceをFalseとして追加
                image_path, image, image_clip, conversations, masks, label, resize, question, sampled_classes = data
                inference = False
            elif isinstance(data, tuple) and len(data) == 10:
                # すでに10要素ある場合はそのまま使用
                image_path, image, image_clip, conversations, masks, label, resize, question, sampled_classes, inference = data
            else:
                # 辞書形式の場合（reason_seg_datasetなど）
                try:
                    if isinstance(data, dict):
                        # 辞書形式からタプル形式に変換
                        image_path = data.get("image_path", "")
                        image = data.get("image", None)
                        image_clip = data.get("image_clip", None)
                        conversations = data.get("conversations", [])
                        masks = data.get("masks", [])
                        label = data.get("label", None)
                        resize = data.get("resize", None)
                        question = data.get("class_names", None)  # reason_segではclass_namesを使用
                        sampled_classes = data.get("class_names", None)
                        inference = data.get("inference", False)
                    else:
                        # 予期しない形式の場合はエラー
                        print(f"警告: 予期しないデータ形式です: {type(data)}, 長さ: {len(data) if isinstance(data, (tuple, list)) else 'N/A'}")
                        raise ValueError(f"予期しないデータ形式: {type(data)}")
                except Exception as e:
                    print(f"エラー: データの処理中に例外が発生しました: {e}")
                    # トレースバックを表示
                    import traceback
                    traceback.print_exc()
                    raise  # 例外を再発生させる
            
            # 最終呼び出し階層の場合はカウンタをリセット
            if call_depth == 1:
                setattr(self, '_call_depth', 0)
                
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
        except Exception as e:
            print(f"エラー: データの取得中に例外が発生しました: {e}")
            import traceback
            traceback.print_exc()
            
            # 再帰呼び出しの代わりに、別のインデックスでもう一度試す
            if call_depth < 5:  # 再帰の深さを制限
                new_idx = random.randint(0, len(self.dataset_list) - 1)
                return self.__getitem__(new_idx)
            else:
                # 再帰が深すぎる場合はダミーデータを返す
                print("警告: データ取得の再試行回数が多すぎます。ダミーデータを返します。")
                setattr(self, '_call_depth', 0)  # カウンタをリセット
                dummy_image = torch.zeros(3, 224, 224)
                dummy_image_clip = torch.zeros(3, 224, 224)
                dummy_masks = torch.zeros(1, 224, 224)
                dummy_label = torch.ones(224, 224) * self.ignore_label
                dummy_conversations = ["データ取得エラー"]
                return (
                    "dummy_path",
                    dummy_image,
                    dummy_image_clip,
                    dummy_conversations,
                    dummy_masks,
                    dummy_label,
                    (224, 224),
                    None,
                    None,
                    False,
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
                    base_image_dir=base_image_dir,
                    tokenizer=tokenizer, 
                    vision_tower=vision_tower,
                    samples_per_epoch=100,
                    precision="fp32",
                    image_size=self.image_size,
                    num_classes_per_sample=1,
                    exclude_val=False,
                    reason_seg_data=f"{val_dataset_name}|{val_split}",
                    explanatory=0.0,
                    processor=processor,
                    val=True,
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
