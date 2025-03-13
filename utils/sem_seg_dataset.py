import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO

from model.llama3_2 import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import ANSWER_LIST, SHORT_QUESTION_LIST


def init_mapillary(base_image_dir):
    try:
        mapillary_data_root = os.path.join(base_image_dir, "mapillary")
        config_path = os.path.join(mapillary_data_root, "config_v2.0.json")
        
        print(f"DEBUG: Checking mapillary config path: {config_path}")
        print(f"DEBUG: Path exists: {os.path.exists(config_path)}")
        
        if not os.path.exists(config_path):
            print("WARNING: mapillary config file not found, returning empty dataset")
            return np.array([]), [], []
        
        with open(config_path) as f:
            mapillary_classes = json.load(f)["labels"]
        mapillary_classes = [x["readable"].lower() for x in mapillary_classes]
        mapillary_classes = np.array(mapillary_classes)
        
        # 小規模データセットの構造では、v2.0/labelsディレクトリが存在しない可能性があります
        # 代わりに直接imagesディレクトリから画像を取得します
        training_images_dir = os.path.join(mapillary_data_root, "training", "images")
        
        print(f"DEBUG: Checking training images directory: {training_images_dir}")
        print(f"DEBUG: Directory exists: {os.path.exists(training_images_dir)}")
        
        if not os.path.exists(training_images_dir):
            print("WARNING: mapillary training images directory not found, returning empty dataset")
            return mapillary_classes, [], []
        
        # 直接JPG画像を取得
        mapillary_images = sorted(glob.glob(os.path.join(training_images_dir, "*.jpg")))
        
        if not mapillary_images:
            print(f"WARNING: No JPG files found in {training_images_dir}")
            print(f"Directory contents: {os.listdir(training_images_dir)}")
            return mapillary_classes, [], []
        
        print(f"DEBUG: Found {len(mapillary_images)} training images")
        
        # 小規模データセットではラベルが含まれていない可能性があるため、空のラベルリストを返します
        # ラベルの代わりにダミー値（None）を使用します
        mapillary_labels = [None] * len(mapillary_images)
        
        print("mapillary: ", len(mapillary_images))
        return mapillary_classes, mapillary_images, mapillary_labels
        
    except Exception as e:
        print(f"ERROR: Failed to initialize mapillary dataset: {e}")
        import traceback
        traceback.print_exc()
        # エラーが発生した場合は空のデータセットを返す
        return np.array([]), [], []


def init_ade20k(base_image_dir):
    try:
        # ADE20Kデータセットのクラス情報を読み込む
        with open("utils/ade20k_classes.json", "r") as f:
            ade20k_classes = json.load(f)
        ade20k_classes = np.array(ade20k_classes)
        
        # パスの構築方法を改善
        # base_image_dirが相対パスか絶対パスかをチェック
        if os.path.isabs(base_image_dir):
            # 絶対パスの場合はそのまま使用
            ade20k_dir = os.path.join(base_image_dir, "ade20k")
        else:
            # 相対パスの場合は現在の作業ディレクトリからの相対パスを構築
            current_dir = os.getcwd()
            print(f"DEBUG: Current working directory: {current_dir}")
            ade20k_dir = os.path.normpath(os.path.join(current_dir, base_image_dir, "ade20k"))
        
        print(f"DEBUG: ADE20K base directory: {ade20k_dir}")
        
        # 画像ディレクトリを確認
        images_dir = os.path.join(ade20k_dir, "images")
        training_dir = os.path.join(images_dir, "training")
        
        print(f"DEBUG: ADE20K images directory: {images_dir}")
        print(f"DEBUG: ADE20K training directory: {training_dir}")
        print(f"DEBUG: Directory exists: {os.path.exists(training_dir)}")
        
        # 訓練用画像のリストを取得
        ade20k_images = []
        ade20k_image_ids = []
        
        if os.path.exists(training_dir):
            # jpg形式のみを検索
            image_files = glob.glob(os.path.join(training_dir, "*.jpg"))
            print(f"DEBUG: Found {len(image_files)} jpg files")
            
            for image_file in image_files:
                image_id = os.path.splitext(os.path.basename(image_file))[0]
                ade20k_images.append(image_file)
                ade20k_image_ids.append(image_id)
        else:
            print(f"WARNING: Training directory does not exist: {training_dir}")
        
        # アノテーションディレクトリを確認
        annotations_dir = os.path.join(ade20k_dir, "annotations", "training")
        if not os.path.exists(annotations_dir):
            print(f"WARNING: [マスクNull原因] Annotations directory does not exist: {annotations_dir}")
            # アノテーションディレクトリが存在しない場合は、代わりにダミーのラベルを返す
            ade20k_labels = [None] * len(ade20k_images)
        else:
            # アノテーションパスの構築
            ade20k_labels = []
            for image_id in ade20k_image_ids:
                label_path = os.path.join(annotations_dir, f"{image_id}.png")
                ade20k_labels.append(label_path)
                # アノテーションファイルの存在を確認
                if not os.path.exists(label_path):
                    print(f"WARNING: [マスクNull原因] Annotation file does not exist: {label_path}")
        
        print("ade20k: ", len(ade20k_images))
        return ade20k_classes, ade20k_images, ade20k_labels
    
    except Exception as e:
        print(f"ERROR: [マスクNull原因] Failed to initialize ade20k dataset: {e}")
        import traceback
        traceback.print_exc()
        # エラーが発生した場合は空のデータセットを返す
        return ade20k_classes, [], []


def init_cocostuff(base_image_dir):
    try:
        cocostuff_classes = []
        with open("utils/cocostuff_classes.txt") as f:
            for line in f.readlines()[1:]:
                cocostuff_classes.append(line.strip().split(": ")[-1])
        cocostuff_classes = np.array(cocostuff_classes)
        
        # 直接ディレクトリからファイルを取得する方法に切り替え
        cocostuff_dir = os.path.join(base_image_dir, "cocostuff", "train2017")
        coco_dir = os.path.join(base_image_dir, "coco", "train2017")
        
        print(f"Looking for annotation files in: {cocostuff_dir}")
        print(f"Looking for image files in: {coco_dir}")
        
        # ディレクトリの存在確認
        if not os.path.exists(cocostuff_dir):
            print(f"WARNING: Cocostuff directory does not exist: {cocostuff_dir}")
            return cocostuff_classes, [], []
            
        if not os.path.exists(coco_dir):
            print(f"WARNING: COCO directory does not exist: {coco_dir}")
            return cocostuff_classes, [], []
            
        # ラベルファイル（PNG）を取得
        cocostuff_labels = sorted(glob.glob(os.path.join(cocostuff_dir, "*.png")))
        if not cocostuff_labels:
            print(f"WARNING: No PNG files found in {cocostuff_dir}")
            return cocostuff_classes, [], []
            
        print(f"Found {len(cocostuff_labels)} label files")
        
        # 対応する画像ファイル（JPG）を構築
        cocostuff_images = []
        valid_labels = []
        
        for label_path in cocostuff_labels:
            # 同じファイル名を持つ画像を探す
            filename = os.path.basename(label_path)
            image_filename = filename.replace(".png", ".jpg")
            image_path = os.path.join(coco_dir, image_filename)
            
            if not os.path.exists(image_path):
                print(f"WARNING: Image file does not exist: {image_path}")
                continue
                
            cocostuff_images.append(image_path)
            valid_labels.append(label_path)
        
        # 有効なラベルパスのみを保持
        cocostuff_labels = valid_labels
        
        print(f"cocostuff: Found {len(cocostuff_images)} valid image-label pairs")
        
        return cocostuff_classes, cocostuff_images, cocostuff_labels
    
    except Exception as e:
        print(f"ERROR: Failed to initialize cocostuff dataset: {e}")
        import traceback
        traceback.print_exc()
        return cocostuff_classes, [], []


def init_paco_lvis(base_image_dir):
    try:
        # 元々期待されているパス
        paco_annotation_path = os.path.join(
            base_image_dir, "vlpart", "paco", "annotations", "paco_lvis_v1_train.json"
        )
        
        # 更新された可能性のあるパス (paco_lvis_v1サブディレクトリあり)
        alt_paco_annotation_path = os.path.join(
            base_image_dir, "vlpart", "paco", "annotations", "paco_lvis_v1", "paco_lvis_v1_train.json"
        )
        
        print(f"DEBUG: Checking paco_lvis original annotation path: {paco_annotation_path}")
        print(f"DEBUG: Original path exists: {os.path.exists(paco_annotation_path)}")
        print(f"DEBUG: Checking paco_lvis alternative annotation path: {alt_paco_annotation_path}")
        print(f"DEBUG: Alternative path exists: {os.path.exists(alt_paco_annotation_path)}")
        
        # どちらかのパスが存在する場合はそれを使用
        if os.path.exists(paco_annotation_path):
            final_path = paco_annotation_path
        elif os.path.exists(alt_paco_annotation_path):
            final_path = alt_paco_annotation_path
            print("INFO: Using alternative path for paco_lvis annotations")
        else:
            # パスが存在しない場合、親ディレクトリの内容をチェック
            paco_dir = os.path.join(base_image_dir, "vlpart", "paco")
            if os.path.exists(paco_dir):
                annotations_dir = os.path.join(paco_dir, "annotations")
                if os.path.exists(annotations_dir):
                    print(f"DEBUG: Annotations directory contents: {os.listdir(annotations_dir)}")
                else:
                    print(f"DEBUG: Annotations directory does not exist: {annotations_dir}")
                    print(f"DEBUG: paco directory contents: {os.listdir(paco_dir)}")
            else:
                print(f"DEBUG: paco directory does not exist: {paco_dir}")
            
            # データセットの小規模版では存在しない可能性があるため、空の結果を返す
            print("WARNING: paco_lvis annotation file not found, returning empty dataset")
            return {}, [], None
            
        coco_api_paco_lvis = COCO(final_path)
        all_classes = coco_api_paco_lvis.loadCats(coco_api_paco_lvis.getCatIds())
        class_map_paco_lvis = {}
        for cat in all_classes:
            cat_split = cat["name"].strip().split(":")
            if len(cat_split) == 1:
                name = cat_split[0].split("_(")[0]
            else:
                assert len(cat_split) == 2
                obj, part = cat_split
                obj = obj.split("_(")[0]
                part = part.split("_(")[0]
                name = (obj, part)
            class_map_paco_lvis[cat["id"]] = name
        img_ids = coco_api_paco_lvis.getImgIds()
        print("paco_lvis: ", len(img_ids))
        return class_map_paco_lvis, img_ids, coco_api_paco_lvis
    
    except Exception as e:
        print(f"ERROR: Failed to initialize paco_lvis dataset: {e}")
        import traceback
        traceback.print_exc()
        # エラーが発生した場合は空のデータセットを返す
        return {}, [], None


def init_pascal_part(base_image_dir):
    try:
        pascal_part_path = os.path.join(base_image_dir, "vlpart", "pascal_part", "train.json")
        
        print(f"DEBUG: Checking pascal_part annotation path: {pascal_part_path}")
        print(f"DEBUG: Path exists: {os.path.exists(pascal_part_path)}")
        
        if not os.path.exists(pascal_part_path):
            print("WARNING: [マスクNull原因] pascal_part annotation file not found, returning empty dataset")
            return {}, [], None
        
        coco_api_pascal_part = COCO(pascal_part_path)
        all_classes = coco_api_pascal_part.loadCats(coco_api_pascal_part.getCatIds())
        class_map_pascal_part = {}
        
        for cat in all_classes:
            cat_name = cat["name"].strip()
            
            # 名前に":"が含まれているかチェック
            if ":" in cat_name:
                # 元の処理：メイン部分とパーツ部分に分割
                cat_main, cat_part = cat_name.split(":")
                name = (cat_main, cat_part)
            else:
                # ":"がない場合は、名前全体をメイン部分とし、パーツ部分を"whole"とする
                print(f"WARNING: Category name '{cat_name}' does not contain ':'. Using whole name as main category.")
                cat_main = cat_name
                cat_part = "whole"
                name = (cat_main, cat_part)
                
            class_map_pascal_part[cat["id"]] = name
            
        img_ids = coco_api_pascal_part.getImgIds()
        print("pascal_part: ", len(img_ids))
        return class_map_pascal_part, img_ids, coco_api_pascal_part
    
    except Exception as e:
        print(f"ERROR: [マスクNull原因] Failed to initialize pascal_part dataset: {e}")
        import traceback
        traceback.print_exc()
        # エラーが発生した場合は空のデータセットを返す
        return {}, [], None


class SemSegDataset(torch.utils.data.Dataset):
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
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
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

        self.data2list = {}
        self.data2classes = {}

        self.sem_seg_datas = sem_seg_data.split("||")
        valid_datasets = []
        
        # 各データセットを初期化し、エラーハンドリングを追加
        for ds in self.sem_seg_datas:
            try:
                print(f"Initializing dataset: {ds}")
                init_func = globals().get(f"init_{ds}")
                if init_func is None:
                    print(f"WARNING: Initialization function 'init_{ds}' not found, skipping")
                    continue
                
                classes, images, labels = init_func(base_image_dir)
                
                # 画像リストが空かどうかチェック
                if not images:
                    print(f"WARNING: No images found for dataset {ds}, skipping")
                    continue
                
                self.data2list[ds] = (images, labels)
                self.data2classes[ds] = classes
                valid_datasets.append(ds)
                print(f"Successfully initialized dataset {ds} with {len(images)} images")
            except Exception as e:
                print(f"ERROR: Failed to initialize dataset {ds}: {e}")
                import traceback
                traceback.print_exc()
        
        # 有効なデータセットがない場合は警告
        if not valid_datasets:
            print("WARNING: No valid datasets were initialized. Training may fail.")
        else:
            print(f"Successfully initialized {len(valid_datasets)} datasets: {', '.join(valid_datasets)}")
        
        # cocoとcocostuffのデータチェック
        if "cocostuff" in self.data2list:
            self.cocostuff_class2index = {
                c: i for i, c in enumerate(self.data2classes["cocostuff"])
            }

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
        # オリジナルLISAコードと同じ方法でデータセットを選択
        ds_idx = random.randint(0, len(self.sem_seg_datas) - 1)
        ds = self.sem_seg_datas[ds_idx]

        # デバッグ情報を出力
        print(f"DEBUG: Selected dataset: {ds}")
        print(f"DEBUG: Available datasets in data2classes: {list(self.data2classes.keys())}")
        
        # データセットが利用可能か確認
        if ds in ["paco_lvis", "pascal_part"] and ds not in self.data2classes:
            print(f"WARNING: Selected dataset {ds} is not available in data2classes. Trying another dataset.")
            # 利用可能なデータセットから選択
            available_datasets = list(self.data2classes.keys())
            if not available_datasets:
                print("ERROR: No valid datasets available!")
                # ダミーデータを返す
                dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
                dummy_image_tensor = torch.from_numpy(dummy_image).permute(2, 0, 1).float()
                dummy_image_clip = torch.zeros(3, 224, 224)
                dummy_masks = torch.zeros(1, 224, 224)
                dummy_label = torch.ones(224, 224) * self.ignore_label
                dummy_conversations = ["No valid datasets available"]
                return (
                    "dummy_path",
                    dummy_image_tensor,
                    dummy_image_clip,
                    dummy_conversations,
                    dummy_masks,
                    dummy_label,
                    (224, 224),
                    None,
                    None,
                )
            
            # 別のデータセットを選択
            ds = random.choice(available_datasets)
            print(f"DEBUG: Selected alternative dataset: {ds}")

        try:
            if ds in ["paco_lvis", "pascal_part"]:
                try:
                    class_map = self.data2classes[ds]
                    img_ids, coco_api = self.data2list[ds]
                    idx = random.randint(0, len(img_ids) - 1)
                    img_id = img_ids[idx]
                    image_info = coco_api.loadImgs([img_id])[0]
                    file_name = image_info["file_name"]
                    if ds == "pascal_part":
                        file_name = os.path.join(
                            "VOCdevkit", "VOC2010", "JPEGImages", file_name
                        )
                        image_path = os.path.join(self.base_image_dir, "vlpart", ds, file_name)
                    elif ds == "paco_lvis":
                        image_path = os.path.join(self.base_image_dir, "coco", file_name)
                    
                    # 画像ファイルの存在チェック
                    if not os.path.exists(image_path):
                        print(f"WARNING: Image file not found: {image_path}")
                        return self.__getitem__(0)
                        
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"WARNING: Failed to load image: {image_path}")
                        return self.__getitem__(0)
                        
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # 画像のプリプロセス
                    image_pil = Image.fromarray(image)
                    processed = self.processor(images=image_pil, return_tensors="pt")
                    image_clip = processed.pixel_values[0]

                    image = self.transform.apply_image(image)  # preprocess image for sam
                    resize = image.shape[:2]
                    annIds = coco_api.getAnnIds(imgIds=image_info["id"])
                    anns = coco_api.loadAnns(annIds)
                    if len(anns) == 0:
                        return self.__getitem__(0)
                    if len(anns) >= self.num_classes_per_sample:
                        sampled_anns = np.random.choice(
                            anns, size=self.num_classes_per_sample, replace=False
                        ).tolist()
                    else:
                        sampled_anns = anns
                    sampled_classes = []
                    for ann in sampled_anns:
                        sampled_cls = class_map[ann["category_id"]]
                        if isinstance(sampled_cls, tuple):
                            obj, part = sampled_cls
                            if random.random() < 0.5:
                                name = obj + " " + part
                            else:
                                name = "the {} of the {}".format(part, obj)
                        else:
                            name = sampled_cls
                        sampled_classes.append(name)

                except KeyError as e:
                    print(f"ERROR: KeyError in dataset {ds}: {e}")
                    return self.__getitem__(0)
                except Exception as e:
                    print(f"ERROR: Unexpected error in dataset {ds}: {e}")
                    return self.__getitem__(0)

            elif ds in ["ade20k", "cocostuff", "mapillary"]:
                try:
                    image, labels = self.data2list[ds]
                    if len(image) == 0:
                        print(f"WARNING: No images available for dataset {ds}")
                        return self.__getitem__(0)
                        
                    idx = random.randint(0, len(image) - 1)
                    image_path = image[idx]
                    label_path = labels[idx]
                    
                    # 画像とラベルの存在チェック
                    if not os.path.exists(image_path):
                        print(f"WARNING: Image file not found: {image_path}")
                        return self.__getitem__(0)
                        
                    if not os.path.exists(label_path):
                        print(f"WARNING: Label file not found: {label_path}")
                        return self.__getitem__(0)
                    
                    # ラベルの読み込み
                    try:
                        label = Image.open(label_path)
                        label = np.array(label)
                    except Exception as e:
                        print(f"WARNING: Failed to load label {label_path}: {e}")
                        return self.__getitem__(0)
                        
                    if ds == "ade20k":
                        label[label == 0] = 255
                        label -= 1
                        label[label == 254] = 255
                    elif ds == "cocostuff":
                        for c, i in self.cocostuff_class2index.items():
                            if "-" in c:
                                label[label == i] = 255
                    
                    # 画像の読み込み
                    try:
                        img = cv2.imread(image_path)
                        if img is None:
                            print(f"WARNING: Failed to load image {image_path}")
                            return self.__getitem__(0)
                        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    except Exception as e:
                        print(f"WARNING: Error loading image {image_path}: {e}")
                        return self.__getitem__(0)

                    # 画像のプリプロセス
                    image_pil = Image.fromarray(image)
                    processed = self.processor(images=image_pil, return_tensors="pt")
                    image_clip = processed.pixel_values[0]

                    image = self.transform.apply_image(image)  # preprocess image for sam
                    resize = image.shape[:2]
                    unique_label = np.unique(label).tolist()
                    if 255 in unique_label:
                        unique_label.remove(255)
                    if len(unique_label) == 0:
                        print(f"WARNING: No valid labels in {label_path}")
                        return self.__getitem__(0)

                    classes = [self.data2classes[ds][class_id] for class_id in unique_label]
                    if len(classes) >= self.num_classes_per_sample:
                        sampled_classes = np.random.choice(
                            classes, size=self.num_classes_per_sample, replace=False
                        ).tolist()
                    else:
                        sampled_classes = classes
                        
                except KeyError as e:
                    print(f"ERROR: KeyError in dataset {ds}: {e}")
                    return self.__getitem__(0)
                except Exception as e:
                    print(f"ERROR: Unexpected error in dataset {ds}: {e}")
                    return self.__getitem__(0)
            else:
                print(f"WARNING: Unknown dataset type: {ds}")
                return self.__getitem__(0)

            questions = []
            answers = []
            class_ids = []
            for sampled_cls in sampled_classes:
                text = sampled_cls

                assert len(text.split("||")) == 1
                question_template = random.choice(self.short_question_list)
                questions.append(question_template.format(class_name=text.lower()))

                answers.append(random.choice(self.answer_list))

                if ds in ["paco_lvis", "pascal_part"]:
                    continue

                class_id = self.data2classes[ds].tolist().index(sampled_cls)
                class_ids.append(class_id)

            conversations = []
            conv = conversation_lib.default_conversation.copy()

            i = 0
            while i < len(questions):
                conv.messages = []
                conv.append_message(conv.roles[0], questions[i])
                conv.append_message(conv.roles[1], answers[i])
                conversations.append(conv.get_prompt())
                i += 1

            image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

            if ds in ["paco_lvis", "pascal_part"]:
                masks = []
                for ann in sampled_anns:
                    try:
                        # アノテーションからマスクを生成する前にセグメンテーション情報があるか確認
                        if "segmentation" not in ann or not ann["segmentation"]:
                            print(f"WARNING: アノテーションにsegmentationフィールドが空またはありません。ID: {ann.get('id', 'unknown')}")
                            # ダミーマスクを生成
                            img_info = coco_api.loadImgs([ann["image_id"]])[0]
                            h, w = img_info["height"], img_info["width"]
                            dummy_mask = np.zeros((h, w), dtype=np.uint8)
                            masks.append(dummy_mask)
                            print(f"  サイズ {h}x{w} のダミーマスクを生成しました")
                            continue
                            
                        # 通常のマスク生成処理
                        mask = coco_api.annToMask(ann)
                        masks.append(mask)
                    except Exception as e:
                        print(f"ERROR: Failed to generate mask: {e}")
                        
                        # エラーのデバッグ情報
                        print(f"  アノテーション情報: {ann.keys() if hasattr(ann, 'keys') else type(ann)}")
                        if hasattr(ann, 'get'):
                            print(f"  アノテーションID: {ann.get('id', 'unknown')}")
                            print(f"  カテゴリID: {ann.get('category_id', 'unknown')}")
                            print(f"  セグメンテーション情報: {type(ann.get('segmentation', None))}")
                            if ann.get('segmentation') is not None:
                                print(f"  セグメンテーションの内容: {ann['segmentation']}")
                        
                        # 代替の空マスクを生成（クラッシュを防ぐ）
                        try:
                            # 画像サイズからダミーマスクを作成
                            if isinstance(image, torch.Tensor):
                                h, w = image.shape[-2], image.shape[-1]
                            else:
                                h, w = image.shape[0], image.shape[1]
                            
                            dummy_mask = np.zeros((h, w), dtype=np.uint8)
                            masks.append(dummy_mask)
                            print(f"  代わりに空マスク（サイズ: {h}x{w}）を使用します")
                        except Exception as mask_e:
                            print(f"  空マスク生成中にもエラーが発生: {mask_e}")
                            # 画像情報からサイズを取得してみる
                            try:
                                img_info = coco_api.loadImgs([ann["image_id"]])[0]
                                h, w = img_info["height"], img_info["width"]
                                dummy_mask = np.zeros((h, w), dtype=np.uint8)
                                masks.append(dummy_mask)
                                print(f"  画像情報から空マスク（サイズ: {h}x{w}）を生成しました")
                            except:
                                # 最小サイズのダミーマスク
                                dummy_mask = np.zeros((100, 100), dtype=np.uint8)
                                masks.append(dummy_mask)
                                print(f"  小さいサイズの空マスク（100x100）を使用します")
                
                # マスクが空の場合は別のサンプルを試す
                if not masks:
                    print("警告: 有効なマスクがありません。別のサンプルを試します。")
                    return self.__getitem__(max(0, idx - 1) if idx > 0 else idx + 1)
                
                try:
                    # マスクをスタック
                    masks = np.stack(masks, axis=0)
                    masks = torch.from_numpy(masks)
                    label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
                except Exception as stack_e:
                    print(f"ERROR: マスクのスタック中にエラーが発生: {stack_e}")
                    print(f"  マスク数: {len(masks)}")
                    if masks:
                        print(f"  最初のマスクの形状: {masks[0].shape if hasattr(masks[0], 'shape') else 'unknown'}")
                    
                    # ダミーのマスクと画像を作成して処理を続行
                    dummy_size = (100, 100)
                    masks = torch.zeros((1, *dummy_size), dtype=torch.uint8)
                    label = torch.ones(dummy_size) * self.ignore_label
                    print(f"  ダミーマスク（サイズ: {dummy_size}）を使用して処理を続行します")

            else:
                label = torch.from_numpy(label).long()
                masks = []
                for class_id in class_ids:
                    masks.append(label == class_id)
                masks = torch.stack(masks, dim=0)
            
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
            
        except Exception as e:
            print(f"ERROR: Unexpected exception in __getitem__: {e}")
            import traceback
            traceback.print_exc()
            return self.__getitem__(0)
