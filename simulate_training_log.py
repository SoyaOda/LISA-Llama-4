#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
トレーニングログをシミュレートするスクリプト
実際の学習は行わず、train_ds.pyの実行時に表示されるログと同様の出力を生成します
"""

import os
import sys
import time
import datetime
import argparse
import glob

def print_separator(message, char="=", length=80):
    print("\n" + char * length)
    print(f"  {message}")
    print(char * length)

def simulate_model_setup():
    print("PEFT適用前のモデル設定確認:")
    print("  - model.config type: MllamaConfig")
    print("  - model.model.config type: MllamaConfig")
    print("  - model.model.config.model_type: mllama")
    print("PEFT適用後のモデル設定確認:")
    print("  - model.base_model.config type: MllamaConfig")
    print("  - model.base_model.config.model_type: mllama")
    print("trainable params: 5,898,240 || all params: 10,791,856,723 || trainable%: 0.05465454324860944")
    print("\n")

def simulate_dataset_warning():
    print_separator("警告: small_test_datasetを使用しています\n"
                   "  このデータセットは限られた画像ファイルのみを含んでいます:\n"
                   "    - ade20k: 10個の画像\n"
                   "    - coco/cocostuff: 10個の画像\n"
                   "    - refer_seg/images/mscoco/images/train2014: 9個の画像\n"
                   "    - reason_seg/ReasonSeg/train: 10組の画像-JSONペア\n"
                   "  指定された`--dataset`と`--refer_seg_data`がこのデータセットと互換性があることを確認してください。")
    print("\n")

def simulate_dataset_initialization(dataset_args):
    datasets = dataset_args.split("||")
    print(f"Dataset: {datasets}")
    
    # SemSegDatasetの初期化
    if "sem_seg" in datasets:
        print("Initializing SemSegDataset with base_dir=./small_test_dataset, data=ade20k||cocostuff||pascal_part||paco_lvis||mapillary")
        
        # ade20k
        print("Initializing dataset: ade20k")
        print("DEBUG: Current working directory: /home/ubuntu/LISA-Llama-4")
        print("DEBUG: ADE20K base directory: /home/ubuntu/LISA-Llama-4/small_test_dataset/ade20k")
        print("DEBUG: ADE20K images directory: /home/ubuntu/LISA-Llama-4/small_test_dataset/ade20k/images")
        print("DEBUG: ADE20K training directory: /home/ubuntu/LISA-Llama-4/small_test_dataset/ade20k/images/training")
        print("DEBUG: Directory exists: True")
        print("DEBUG: Found 10 jpg files")
        print("ade20k:  10")
        print("Successfully initialized dataset ade20k with 10 images")
        
        # cocostuff
        print("Initializing dataset: cocostuff")
        print("Found cocostuff training directory: /home/ubuntu/LISA-Llama-4/small_test_dataset/cocostuff/train2017")
        print("Found coco training directory: /home/ubuntu/LISA-Llama-4/small_test_dataset/coco/train2017")
        print("DEBUG: Current working directory: /home/ubuntu/LISA-Llama-4")
        print("DEBUG: Cocostuff directory: /home/ubuntu/LISA-Llama-4/small_test_dataset/cocostuff")
        print("DEBUG: Cocostuff train directory: /home/ubuntu/LISA-Llama-4/small_test_dataset/cocostuff/train2017")
        print("DEBUG: Cocostuff train directory exists: True")
        print("DEBUG: Coco directory: /home/ubuntu/LISA-Llama-4/small_test_dataset/coco")
        print("DEBUG: Coco train directory: /home/ubuntu/LISA-Llama-4/small_test_dataset/coco/train2017")
        print("DEBUG: Coco train directory exists: True")
        print("DEBUG: Found 10 PNG files in /home/ubuntu/LISA-Llama-4/small_test_dataset/cocostuff/train2017")
        print("DEBUG: Found 10 label files")
        print("cocostuff:  10")
        print("Successfully initialized dataset cocostuff with 10 images")
        
        # pascal_part
        print("Initializing dataset: pascal_part")
        print("DEBUG: Checking pascal_part annotation path: ./small_test_dataset/vlpart/pascal_part/train.json")
        print("DEBUG: Path exists: True")
        print("loading annotations into memory...")
        print("Done (t=0.00s)")
        print("creating index...")
        print("index created!")
        
        # 警告メッセージの出力
        for category in ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
                         "chair", "cow", "dog", "horse", "motorbike", "person", "pottedplant", 
                         "sheep", "sofa", "table", "train", "tvmonitor"]:
            print(f"WARNING: Category name '{category}' does not contain ':'. Using whole name as main category.")
        
        print("pascal_part:  10")
        print("Successfully initialized dataset pascal_part with 10 images")
        
        # paco_lvis
        print("Initializing dataset: paco_lvis")
        print("DEBUG: Checking paco_lvis original annotation path: ./small_test_dataset/vlpart/paco/annotations/paco_lvis_v1_train.json")
        print("DEBUG: Original path exists: False")
        print("DEBUG: Checking paco_lvis alternative annotation path: ./small_test_dataset/vlpart/paco/annotations/paco_lvis_v1/paco_lvis_v1_train.json")
        print("DEBUG: Alternative path exists: True")
        print("INFO: Using alternative path for paco_lvis annotations")
        print("loading annotations into memory...")
        print("Done (t=0.00s)")
        print("creating index...")
        print("index created!")
        print("paco_lvis:  10")
        print("Successfully initialized dataset paco_lvis with 10 images")
        
        # mapillary
        print("Initializing dataset: mapillary")
        print("DEBUG: Checking mapillary config path: ./small_test_dataset/mapillary/config_v2.0.json")
        print("DEBUG: Path exists: True")
        print("DEBUG: Checking training images directory: ./small_test_dataset/mapillary/training/images")
        print("DEBUG: Directory exists: True")
        print("DEBUG: Found 10 training images")
        print("mapillary:  10")
        print("Successfully initialized dataset mapillary with 10 images")
        
        print("Successfully initialized 5 datasets: ade20k, cocostuff, pascal_part, paco_lvis, mapillary")
    
    # refer_segデータセットの確認と初期化
    if "refer_seg" in datasets:
        print("確認: refclefデータセットが利用可能 (./small_test_dataset/refer_seg/refclef/refclef/refs(unc).p)")
        print("確認: refcocoデータセットが利用可能 (./small_test_dataset/refer_seg/refcoco/refcoco/refs(unc).p)")
        print("確認: refcoco+データセットが利用可能 (./small_test_dataset/refer_seg/refcoco+/refcoco+/refs(unc).p)")
        print("エラー: refcocogのrefファイルが見つかりません: ./small_test_dataset/refer_seg/refcocog/refcocog/refs(unc).p")
        print("利用可能なrefer_segデータセット: refclef||refcoco||refcoco+")
        print("ReferSegDatasetを初期化します: base_dir=./small_test_dataset, data=refclef||refcoco||refcoco+")
        
        # refclef
        print("初期化: REFER data_root=./small_test_dataset/refer_seg, dataset=refclef, splitBy=unc")
        print("代替パスを確認: ./small_test_dataset/refer_seg/refclef/refclef")
        print("参照ファイル: ./small_test_dataset/refer_seg/refclef/refclef/refs(unc).p")
        print("ファイル存在: True")
        print("利用可能な画像ファイル数: 10")
        print("フィルタリング後のアノテーション数: 294 (元: 294)")
        print("フィルタリング後の参照数: 293 (元: 293)")
        print("creating index...")
        print("index created.")
        print("完了 (t=0.00s)")
        print("dataset refclef (refs unc) (train split) has 7 images and 294 annotations.")
        
        # refcoco（エラーが発生するデータセット）
        print("初期化: REFER data_root=./small_test_dataset/refer_seg, dataset=refcoco, splitBy=unc")
        print("代替パスを確認: ./small_test_dataset/refer_seg/refcoco/refcoco")
        print("参照ファイル: ./small_test_dataset/refer_seg/refcoco/refcoco/refs(unc).p")
        print("ファイル存在: True")
        print("利用可能な画像ファイル数: 10")
        print("フィルタリング後の画像数: 0 (元: 10)")
        print("フィルタリング後のアノテーション数: 0 (元: 153)")
        print("フィルタリング後の参照数: 0 (元: 107)")
        print("creating index...")
        print("index created.")
        print("完了 (t=0.00s)")
        
        # エラーメッセージの出力
        print_separator("エラー: データセットの初期化に失敗しました\n"
                       "原因: 'NoneType' object is not iterable\n"
                       "解決策:\n"
                       "1. データセットの構造とコード実装の互換性を確認してください\n"
                       "2. `--dataset`および関連するデータセット引数を確認してください")
        
        # トレースバックの出力
        print("\n\nTraceback (most recent call last):")
        print("  File \"/home/ubuntu/LISA-Llama-4/train_ds.py\", line 755, in <module>")
        print("    main(sys.argv[1:])")
        print("  File \"/home/ubuntu/LISA-Llama-4/train_ds.py\", line 322, in main")
        print("    train_dataset = HybridDataset(")
        print("  File \"/home/ubuntu/LISA-Llama-4/utils/dataset.py\", line 385, in __init__")
        print("    self.refer_seg_dataset = ReferSegDataset(")
        print("  File \"/home/ubuntu/LISA-Llama-4/utils/refer_seg_dataset.py\", line 78, in __init__")
        print("    for item in loaded_images:")
        print("TypeError: 'NoneType' object is not iterable")
        
        # Deepspeed関連の出力
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time},097] [INFO] [launch.py:319:sigkill_handler] Killing subprocess 56352")
        print(f"[{current_time},097] [ERROR] [launch.py:325:sigkill_handler] ['/home/ubuntu/LISA-Llama-4/lisa_env/bin/python', '-u', 'train_ds.py', '--local_rank=0', '--version=meta-llama/Llama-3.2-11B-Vision-Instruct', '--dataset_dir=./small_test_dataset', '--vision_pretrained=./checkpoints/sam_vit_h_4b8939.pth', '--vision-tower=meta-llama/Llama-3.2-11B-Vision-Instruct', '--dataset=sem_seg||refer_seg||vqa||reason_seg', '--sample_rates=9,3,3,1', '--exp_name=lisa-llama3-2-vision-test', '--conv_type=llama_3', '--precision=bf16', '--batch_size=1', '--grad_accumulation_steps=1', '--steps_per_epoch=2', '--epochs=1', '--no_eval'] exits with return code = 1")
        
        # 処理の終了
        sys.exit(1)
    
    # vqaとreason_segは初期化されない（refer_segでエラーが発生して終了するため）
    if "vqa" in datasets:
        print("Initializing VQADataset...")
    
    if "reason_seg" in datasets:
        print("Initializing ReasonSegDataset...")

def main():
    parser = argparse.ArgumentParser(description="トレーニングログをシミュレートする")
    parser.add_argument("--dataset", type=str, default="sem_seg||refer_seg||vqa||reason_seg",
                        help="使用するデータセット（||区切り）")
    parser.add_argument("--version", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct",
                        help="使用するモデルバージョン")
    parser.add_argument("--dataset_dir", type=str, default="./small_test_dataset",
                        help="データセットディレクトリ")
    parser.add_argument("--sample_rates", type=str, default="9,3,3,1",
                        help="サンプルレート")
    
    args = parser.parse_args()
    
    # モデルのセットアップをシミュレート
    simulate_model_setup()
    
    # small_test_datasetに関する警告
    simulate_dataset_warning()
    
    # データセットの初期化をシミュレート
    simulate_dataset_initialization(args.dataset)
    
    # ここまで到達することはない（refer_segでエラーが発生して終了するため）
    print("Training started...")

if __name__ == "__main__":
    main() 