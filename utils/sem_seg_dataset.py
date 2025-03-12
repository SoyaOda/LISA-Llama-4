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
        ade20k_classes = []
        with open("utils/ade20k_classes.txt") as f:
            for line in f.readlines():
                ade20k_classes.append(line.strip())
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
        
        # パスの構築方法を改善
        # base_image_dirが相対パスか絶対パスかをチェック
        if os.path.isabs(base_image_dir):
            # 絶対パスの場合はそのまま使用
            cocostuff_dir = os.path.join(base_image_dir, "cocostuff")
            coco_dir = os.path.join(base_image_dir, "coco")
        else:
            # 相対パスの場合は現在の作業ディレクトリからの相対パスを構築
            current_dir = os.getcwd()
            cocostuff_dir = os.path.normpath(os.path.join(current_dir, base_image_dir, "cocostuff"))
            coco_dir = os.path.normpath(os.path.join(current_dir, base_image_dir, "coco"))
        
        # 可能性のあるディレクトリパスを確認
        possible_cocostuff_dirs = [
            os.path.join(cocostuff_dir, "train2017"),
            os.path.join(base_image_dir, "cocostuff/train2017"),
            os.path.join(cocostuff_dir)
        ]
        
        cocostuff_train_dir = None
        for dir_path in possible_cocostuff_dirs:
            if os.path.exists(dir_path):
                print(f"Found cocostuff training directory: {dir_path}")
                cocostuff_train_dir = dir_path
                break
        
        # 見つからない場合は最初のパスを使用
        if not cocostuff_train_dir:
            cocostuff_train_dir = possible_cocostuff_dirs[0]
            print(f"WARNING: [マスクNull原因] Could not find cocostuff directory, using default: {cocostuff_train_dir}")
        
        # cocoのトレーニングディレクトリも同様に検索
        possible_coco_dirs = [
            os.path.join(coco_dir, "train2017"),
            os.path.join(base_image_dir, "coco/train2017"),
            os.path.join(coco_dir)
        ]
        
        coco_train_dir = None
        for dir_path in possible_coco_dirs:
            if os.path.exists(dir_path):
                print(f"Found coco training directory: {dir_path}")
                coco_train_dir = dir_path
                break
        
        # 見つからない場合は最初のパスを使用
        if not coco_train_dir:
            coco_train_dir = possible_coco_dirs[0]
            print(f"WARNING: [マスクNull原因] Could not find coco directory, using default: {coco_train_dir}")
        
        print(f"DEBUG: Current working directory: {os.getcwd()}")
        print(f"DEBUG: Cocostuff directory: {cocostuff_dir}")
        print(f"DEBUG: Cocostuff train directory: {cocostuff_train_dir}")
        print(f"DEBUG: Cocostuff train directory exists: {os.path.exists(cocostuff_train_dir)}")
        print(f"DEBUG: Coco directory: {coco_dir}")
        print(f"DEBUG: Coco train directory: {coco_train_dir}")
        print(f"DEBUG: Coco train directory exists: {os.path.exists(coco_train_dir)}")
        
        # ディレクトリが存在しない場合は空のリストを返す
        if not os.path.exists(cocostuff_train_dir):
            print(f"WARNING: [マスクNull原因] Cocostuff train directory not found: {cocostuff_train_dir}")
            return cocostuff_classes, [], []
            
        if not os.path.exists(coco_train_dir):
            print(f"WARNING: [マスクNull原因] Coco train directory not found: {coco_train_dir}")
            return cocostuff_classes, [], []
        
        # ラベルファイル（PNG）を取得
        cocostuff_labels = []
        
        # まずディレクトリ内のすべてのPNGファイルを検索
        try:
            png_files = glob.glob(os.path.join(cocostuff_train_dir, "*.png"))
            if png_files:
                cocostuff_labels = png_files
                print(f"DEBUG: Found {len(png_files)} PNG files in {cocostuff_train_dir}")
            else:
                print(f"WARNING: [マスクNull原因] No PNG files found in {cocostuff_train_dir}")
                if os.path.exists(cocostuff_train_dir):
                    print(f"Directory contents: {os.listdir(cocostuff_train_dir)}")
                    
                    # もしPNGファイルがなければ、代わりにJPGファイルを探す
                    jpg_files = glob.glob(os.path.join(cocostuff_train_dir, "*.jpg"))
                    if jpg_files:
                        print(f"Found {len(jpg_files)} JPG files instead, using these")
                        cocostuff_labels = jpg_files
                
        except Exception as e:
            print(f"ERROR: [マスクNull原因] Error searching for PNG files: {e}")
            import traceback
            traceback.print_exc()
        
        if not cocostuff_labels:
            print(f"WARNING: [マスクNull原因] No label files found for cocostuff in {cocostuff_train_dir}")
            return cocostuff_classes, [], []
        
        cocostuff_labels = sorted(cocostuff_labels)
        print(f"DEBUG: Found {len(cocostuff_labels)} label files")
        
        # 対応する画像ファイル（JPG）のパスを構築
        cocostuff_images = []
        valid_pairs = []
        for label_path in cocostuff_labels:
            # ファイル名のみを取得
            filename = os.path.basename(label_path)
            image_filename = filename.replace(".png", ".jpg")
            image_path = os.path.join(coco_train_dir, image_filename)
            
            # 画像ファイルの存在を確認
            if not os.path.exists(image_path):
                print(f"WARNING: [マスクNull原因] Image file does not exist: {image_path}")
                # 可能性のあるパスを試す
                alternative_paths = [
                    os.path.join(cocostuff_dir, image_filename),
                    os.path.join(coco_dir, image_filename),
                    os.path.join(base_image_dir, "coco", image_filename)
                ]
                
                found_alternative = False
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        print(f"Found alternative path for image: {alt_path}")
                        image_path = alt_path
                        found_alternative = True
                        break
                
                if not found_alternative:
                    print(f"WARNING: [マスクNull原因] Could not find image for label: {label_path}")
                    continue
            
            cocostuff_images.append(image_path)
            valid_pairs.append(label_path)
        
        # 有効なラベルパスのみを保持
        cocostuff_labels = valid_pairs
        
        print("cocostuff: ", len(cocostuff_images))
        if len(cocostuff_images) != len(cocostuff_labels):
            print(f"WARNING: [マスクNull原因] Mismatch between images ({len(cocostuff_images)}) and labels ({len(cocostuff_labels)})")
        
        return cocostuff_classes, cocostuff_images, cocostuff_labels
    
    except Exception as e:
        print(f"ERROR: [マスクNull原因] Failed to initialize cocostuff dataset: {e}")
        import traceback
        traceback.print_exc()
        # 例外を再発生させる代わりに空のリストを返す
        print("Returning empty lists for cocostuff")
        return cocostuff_classes, [], []


def init_paco_lvis(base_image_dir):
    try:
        paco_annotation_path = os.path.join(
            base_image_dir, "vlpart", "paco", "annotations", "paco_lvis_v1_train.json"
        )
        
        print(f"DEBUG: Checking paco_lvis annotation path: {paco_annotation_path}")
        print(f"DEBUG: Path exists: {os.path.exists(paco_annotation_path)}")
        
        # パスが存在しない場合、親ディレクトリの内容をチェック
        if not os.path.exists(paco_annotation_path):
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
            
        coco_api_paco_lvis = COCO(paco_annotation_path)
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
            print("WARNING: pascal_part annotation file not found, returning empty dataset")
            return {}, [], None
        
        coco_api_pascal_part = COCO(pascal_part_path)
        all_classes = coco_api_pascal_part.loadCats(coco_api_pascal_part.getCatIds())
        class_map_pascal_part = {}
        for cat in all_classes:
            cat_main, cat_part = cat["name"].strip().split(":")
            name = (cat_main, cat_part)
            class_map_pascal_part[cat["id"]] = name
        img_ids = coco_api_pascal_part.getImgIds()
        print("pascal_part: ", len(img_ids))
        return class_map_pascal_part, img_ids, coco_api_pascal_part
    
    except Exception as e:
        print(f"ERROR: Failed to initialize pascal_part dataset: {e}")
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
                        masks.append(coco_api.annToMask(ann))
                    except Exception as e:
                        print(f"ERROR: Failed to generate mask: {e}")
                        return self.__getitem__(0)

                masks = np.stack(masks, axis=0)
                masks = torch.from_numpy(masks)
                label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

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
