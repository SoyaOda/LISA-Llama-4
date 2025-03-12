import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from PIL import Image

from model.llama3_2 import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .grefer import G_REFER
from .refer import REFER
from .utils import ANSWER_LIST, SHORT_QUESTION_LIST


class ReferSegDataset(torch.utils.data.Dataset):
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
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
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

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        DATA_DIR = os.path.join(base_image_dir, "refer_seg")
        self.refer_seg_ds_list = refer_seg_data.split(
            "||"
        )  # ['refclef', 'refcoco', 'refcoco+', 'refcocog']
        self.refer_seg_data = {}
        for ds in self.refer_seg_ds_list:
            if ds == "refcocog":
                splitBy = "umd"
            else:
                splitBy = "unc"

            if ds == "grefcoco":
                refer_api = G_REFER(DATA_DIR, ds, splitBy)
            else:
                refer_api = REFER(DATA_DIR, ds, splitBy)
            ref_ids_train = refer_api.getRefIds(split="train")
            images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)

            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_train)

            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/saiapr_tc-12", item["file_name"]
                    )
                else:
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/mscoco/images/train2014", item["file_name"]
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_train

            print(
                "dataset {} (refs {}) (train split) has {} images and {} annotations.".format(
                    ds,
                    splitBy,
                    len(refer_seg_ds["images"]),
                    len(refer_seg_ds["annotations"]),
                )
            )

            img2refs = {}
            for ref in refs_train:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_data[ds] = refer_seg_ds

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
        # 以前の複雑な再帰呼び出しをシンプルなエラー処理に変更
        max_attempts = 20  # 最大試行回数
        for attempt in range(max_attempts):
            try:
                # ランダムにデータセットを選択
                ds_idx = random.randint(0, len(self.refer_seg_ds_list) - 1)
                ds = self.refer_seg_ds_list[ds_idx]
                refer_seg_ds = self.refer_seg_data[ds]
                
                images = refer_seg_ds["images"]
                annotations = refer_seg_ds["annotations"]
                img2refs = refer_seg_ds["img2refs"]
                
                # ランダムに画像を選択
                img_idx = random.randint(0, len(images) - 1)
                image_info = images[img_idx]
                image_path = image_info["file_name"]
                
                # 画像ファイルの存在確認
                if not os.path.exists(image_path):
                    print(f"WARNING: Image not found: {image_path}")
                    continue  # 次の試行へ
                
                image_id = image_info["id"]
                refs = img2refs.get(image_id, [])
                
                if len(refs) == 0:
                    print(f"WARNING: No references for image_id {image_id}")
                    continue  # 次の試行へ
                
                # テキストとアノテーションIDの収集
                sents = []
                ann_ids = []
                for ref in refs:
                    for sent in ref["sentences"]:
                        text = sent["sent"]
                        sents.append(text)
                        ann_ids.append(ref["ann_id"])
                
                # サンプル数の調整
                if len(sents) >= self.num_classes_per_sample:
                    sampled_inds = np.random.choice(
                        list(range(len(sents))), size=self.num_classes_per_sample, replace=False
                    )
                else:
                    sampled_inds = list(range(len(sents)))
                
                sampled_sents = [sents[i] for i in sampled_inds]
                sampled_ann_ids = [ann_ids[i] for i in sampled_inds]
                sampled_classes = sampled_sents
                
                # 画像の読み込み
                image = cv2.imread(image_path)
                if image is None:
                    print(f"WARNING: Failed to load image {image_path}")
                    continue  # 次の試行へ
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 画像のプリプロセス
                image_pil = Image.fromarray(image)
                # プロセッサを使用してLlama3 Vision用の入力を作成
                processed = self.processor(images=image_pil, return_tensors="pt")
                image_clip = processed.pixel_values[0]
                
                # SAM用の画像前処理
                image_transformed = self.transform.apply_image(image)
                resize = image_transformed.shape[:2]
                
                # 質問と回答の作成
                questions = []
                answers = []
                for text in sampled_classes:
                    text = text.strip()
                    if len(text.split("||")) > 1:
                        # クラス名が複雑な場合の処理
                        text = text.split("||")[0]
                    
                    question_template = random.choice(self.short_question_list)
                    questions.append(question_template.format(class_name=text.lower()))
                    answers.append(random.choice(self.answer_list))
                
                # 会話形式の作成
                conversations = []
                conv = conversation_lib.default_conversation.copy()
                
                for i in range(len(questions)):
                    conv.messages = []
                    conv.append_message(conv.roles[0], questions[i])
                    conv.append_message(conv.roles[1], answers[i])
                    conversations.append(conv.get_prompt())
                
                # 画像のテンソル変換
                image_tensor = self.preprocess(torch.from_numpy(image_transformed).permute(2, 0, 1).contiguous())
                
                # マスクの処理
                masks = []
                for ann_id in sampled_ann_ids:
                    if isinstance(ann_id, list):
                        # 複数アノテーションの場合
                        m_final = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
                        for ann_id_i in ann_id:
                            try:
                                ann = annotations[ann_id_i]
                                if len(ann["segmentation"]) == 0:
                                    m = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
                                else:
                                    m = mask.decode(ann["segmentation"])
                                m_final = np.logical_or(m_final, m)
                            except:
                                print(f"WARNING: Error processing annotation {ann_id_i}")
                        
                        m_final = m_final.astype(np.uint8)
                    else:
                        # 単一アノテーションの場合
                        try:
                            ann = annotations[ann_id]
                            if len(ann["segmentation"]) == 0:
                                m_final = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
                            else:
                                m_final = mask.decode(ann["segmentation"])
                        except:
                            print(f"WARNING: Error processing annotation {ann_id}")
                            m_final = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
                    
                    # マスクのリサイズと追加
                    m_final = cv2.resize(
                        m_final, (image_tensor.shape[2], image_tensor.shape[1]), interpolation=cv2.INTER_NEAREST
                    )
                    masks.append(torch.from_numpy(m_final).float())
                
                if not masks:
                    # マスクが作成できなかった場合
                    print(f"WARNING: Failed to create any masks")
                    continue  # 次の試行へ
                
                masks = torch.stack(masks, dim=0)
                
                # SAM用のラベル作成
                h, w = resize
                label = np.ones((h, w)) * self.ignore_label
                
                # 正常に処理できた場合、結果を返す
                return (
                    image_path,
                    image_tensor,
                    image_clip,
                    conversations,
                    masks,
                    torch.from_numpy(label).long(),
                    resize,
                    questions,
                    sampled_classes,
                    False,  # inference flag
                )
                
            except Exception as e:
                print(f"ERROR in refer_seg_dataset.__getitem__: {e}")
                continue  # 次の試行へ
        
        # すべての試行が失敗した場合、ダミーデータを返す
        print(f"WARNING: All {max_attempts} attempts failed in refer_seg_dataset.__getitem__, returning dummy data")
        dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
        dummy_image_tensor = torch.from_numpy(dummy_image).permute(2, 0, 1).float()
        dummy_image_clip = torch.zeros(3, 224, 224)
        dummy_masks = torch.zeros(1, 224, 224)
        dummy_label = torch.ones(224, 224) * self.ignore_label
        dummy_conversations = ["No valid data available"]
        
        return (
            "dummy_path",
            dummy_image_tensor,
            dummy_image_clip,
            dummy_conversations,
            dummy_masks,
            torch.from_numpy(dummy_label).long(),
            (224, 224),
            None,
            None,
            False,  # inference flag
        )
