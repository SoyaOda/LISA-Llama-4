#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
small_test_datasetの構造チェックスクリプト
データセットの構造、ファイルの有無、IDの整合性などを検証します
"""

import os
import glob
import json
import pickle
import sys
from collections import defaultdict

# 色付きの出力のための定数
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
ENDC = '\033[0m'

def print_colored(text, color):
    """色付きテキストを出力する"""
    print(f"{color}{text}{ENDC}")

def print_section(title):
    """セクションタイトルを出力する"""
    print("\n" + "=" * 80)
    print_colored(f"  {title}", BLUE)
    print("=" * 80)

def count_files(directory, pattern):
    """指定したパターンに一致するファイルの数を数える"""
    return len(glob.glob(os.path.join(directory, pattern), recursive=True))

def get_file_size(file_path):
    """ファイルサイズを取得する（人間が読みやすい形式で）"""
    if not os.path.exists(file_path):
        return "ファイルが存在しません"
    
    size_bytes = os.path.getsize(file_path)
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes/(1024*1024):.2f} MB"
    else:
        return f"{size_bytes/(1024*1024*1024):.2f} GB"

def load_pickle_file(file_path):
    """pickleファイルを読み込む"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            return data
    except Exception as e:
        print_colored(f"エラー: {file_path} の読み込み中にエラーが発生しました: {e}", RED)
        return None

def load_json_file(file_path):
    """JSONファイルを読み込む"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print_colored(f"エラー: {file_path} の読み込み中にエラーが発生しました: {e}", RED)
        return None

def check_directory_structure(base_dir):
    """ディレクトリ構造を確認する"""
    print_section("ディレクトリ構造の概要")
    
    if not os.path.exists(base_dir):
        print_colored(f"エラー: {base_dir} が存在しません", RED)
        return False
    
    for root, dirs, files in os.walk(base_dir):
        level = root.replace(base_dir, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        
        # ディレクトリ内のファイル数のサマリーを表示
        if len(files) > 10:
            print(f"{indent}    ... {len(files)} ファイル")
        else:
            for f in files:
                if f.startswith('.'):  # 隠しファイルはスキップ
                    continue
                print(f"{indent}    {f}")
    
    return True

def check_refcoco_datasets(base_dir):
    """refer_segデータセットの詳細をチェックする"""
    print_section("refer_segデータセットの分析")
    
    refer_seg_dir = os.path.join(base_dir, "refer_seg")
    if not os.path.exists(refer_seg_dir):
        print_colored(f"エラー: {refer_seg_dir} が存在しません", RED)
        return False
    
    # 1. 画像ファイルの確認
    print_colored("画像ファイルの確認:", BLUE)
    mscoco_img_dir = os.path.join(refer_seg_dir, "images", "mscoco", "images", "train2014")
    if os.path.exists(mscoco_img_dir):
        images = glob.glob(os.path.join(mscoco_img_dir, "*.jpg"))
        print(f"MSCOCO画像ファイル数: {len(images)}")
        
        # サンプルの表示
        if images:
            print("サンプル画像ファイル:")
            for img in images[:5]:
                print(f"  - {os.path.basename(img)}")
            if len(images) > 5:
                print(f"  - ... 他 {len(images)-5} ファイル")
        
        # 画像IDの抽出
        img_ids = []
        for img in images:
            basename = os.path.basename(img)
            if basename.startswith("COCO_train2014_"):
                # COCO_train2014_000000000009.jpg → 000000000009
                img_id = basename.split("_")[2].split(".")[0]
                img_ids.append(img_id)
        
        print(f"抽出された画像ID数: {len(img_ids)}")
        if img_ids:
            print("サンプル画像ID:")
            for id in img_ids[:5]:
                print(f"  - {id}")
    else:
        print_colored(f"警告: MSCOCO画像ディレクトリ {mscoco_img_dir} が存在しません", YELLOW)
    
    # 2. アノテーションファイルの確認
    print_colored("\nアノテーションファイルの確認:", BLUE)
    datasets = ["refcoco", "refcoco+", "refcocog", "refclef"]
    
    for dataset in datasets:
        dataset_dir = os.path.join(refer_seg_dir, dataset, dataset)
        if not os.path.exists(dataset_dir):
            print(f"{dataset}: ディレクトリが存在しません")
            continue
        
        # JSONファイルの確認
        json_path = os.path.join(dataset_dir, "instances.json")
        json_size = get_file_size(json_path)
        print(f"{dataset} instances.json: {json_size}")
        
        # Pickleファイルの確認
        p_files = glob.glob(os.path.join(dataset_dir, "refs*.p"))
        print(f"{dataset} Pickleファイル: {len(p_files)}個")
        
        for p_file in p_files:
            p_size = get_file_size(p_file)
            print(f"  - {os.path.basename(p_file)}: {p_size}")
        
        # JSONファイルの内容確認
        if os.path.exists(json_path):
            print(f"\n{dataset} JSONファイルの分析:")
            json_data = load_json_file(json_path)
            
            if json_data:
                if 'images' in json_data:
                    print(f"  画像エントリ数: {len(json_data['images'])}")
                    
                    # サンプル画像IDの確認
                    if img_ids and len(json_data['images']) > 0:
                        matches = 0
                        for entry in json_data['images']:
                            if 'id' in entry and str(entry['id']) in img_ids:
                                matches += 1
                        
                        match_percent = (matches / len(img_ids)) * 100 if img_ids else 0
                        if match_percent > 0:
                            print_colored(f"  画像IDの一致率: {match_percent:.2f}% ({matches}/{len(img_ids)})", GREEN)
                        else:
                            print_colored(f"  画像IDの一致率: {match_percent:.2f}% ({matches}/{len(img_ids)})", RED)
                            print_colored("  警告: JSONファイルの画像IDとサンプル画像のIDが一致していません", RED)
                
                if 'annotations' in json_data:
                    print(f"  アノテーション数: {len(json_data['annotations'])}")
        
        # Pickleファイルの内容確認（サンプル）
        if p_files:
            print(f"\n{dataset} Pickleファイルの分析:")
            try:
                sample_p_file = p_files[0]
                p_data = load_pickle_file(sample_p_file)
                
                if p_data:
                    if isinstance(p_data, dict) and 'refs' in p_data:
                        # 辞書形式のPickleファイル処理（元の想定形式）
                        print(f"  参照エントリ数: {len(p_data['refs'])}")
                        
                        # サンプル参照の表示
                        if len(p_data['refs']) > 0:
                            sample_ref = p_data['refs'][0]
                            print("  サンプル参照エントリ:")
                            for key, value in sample_ref.items():
                                if key not in ['sentences']:
                                    print(f"    - {key}: {value}")
                            
                            # 画像IDの整合性チェック
                            ref_img_ids = set()
                            for ref in p_data['refs']:
                                if 'image_id' in ref:
                                    ref_img_ids.add(str(ref['image_id']))
                            
                            # 実際の画像IDとの比較
                            if img_ids:
                                matches = len(set(img_ids) & ref_img_ids)
                                match_percent = (matches / len(img_ids)) * 100 if img_ids else 0
                                
                                if match_percent > 0:
                                    print_colored(f"  画像IDの一致率: {match_percent:.2f}% ({matches}/{len(img_ids)})", GREEN)
                                else:
                                    print_colored(f"  画像IDの一致率: {match_percent:.2f}% ({matches}/{len(img_ids)})", RED)
                                    print_colored("  警告: Pickleファイルの画像IDとサンプル画像のIDが一致していません", RED)
                                    
                                # 実際のIDのサンプル表示
                                print("  Pickle内の画像IDサンプル:")
                                for id in list(ref_img_ids)[:5]:
                                    print(f"    - {id}")
                    elif isinstance(p_data, list):
                        # リスト形式のPickleファイル処理
                        print(f"  参照エントリ数: {len(p_data)}")
                        
                        # サンプル参照の表示
                        if len(p_data) > 0:
                            sample_ref = p_data[0]
                            print("  サンプル参照エントリ:")
                            if isinstance(sample_ref, dict):
                                for key, value in sample_ref.items():
                                    if key not in ['sentences']:
                                        print(f"    - {key}: {value}")
                            else:
                                print(f"    - データ型: {type(sample_ref)}")
                            
                            # 画像IDの整合性チェック
                            ref_img_ids = set()
                            for ref in p_data:
                                if isinstance(ref, dict) and 'image_id' in ref:
                                    ref_img_ids.add(str(ref['image_id']))
                            
                            # 実際の画像IDとの比較
                            if img_ids and ref_img_ids:
                                matches = len(set(img_ids) & ref_img_ids)
                                match_percent = (matches / len(img_ids)) * 100 if img_ids else 0
                                
                                if match_percent > 0:
                                    print_colored(f"  画像IDの一致率: {match_percent:.2f}% ({matches}/{len(img_ids)})", GREEN)
                                else:
                                    print_colored(f"  画像IDの一致率: {match_percent:.2f}% ({matches}/{len(img_ids)})", RED)
                                    print_colored("  警告: Pickleファイルの画像IDとサンプル画像のIDが一致していません", RED)
                                
                                # 実際のIDのサンプル表示
                                print("  Pickle内の画像IDサンプル:")
                                for id in list(ref_img_ids)[:5]:
                                    print(f"    - {id}")
                    else:
                        print_colored(f"  警告: Pickleファイルの形式が想定と異なります。", YELLOW)
                        print(f"  Pickleファイルの内容: {type(p_data)}")
            except Exception as e:
                print_colored(f"  エラー: Pickleファイルの分析中に例外が発生しました: {e}", RED)
    
    return True

def check_other_datasets(base_dir):
    """その他のデータセットをチェックする"""
    print_section("その他のデータセットの分析")
    
    datasets = ["sem_seg", "vqa", "reason_seg"]
    
    for dataset in datasets:
        dataset_dir = os.path.join(base_dir, dataset)
        if not os.path.exists(dataset_dir):
            print_colored(f"{dataset}: ディレクトリが存在しません", YELLOW)
            continue
        
        print_colored(f"{dataset}の確認:", BLUE)
        
        # 画像ファイル数の確認
        img_count = count_files(dataset_dir, "**/*.jpg")
        png_count = count_files(dataset_dir, "**/*.png")
        json_count = count_files(dataset_dir, "**/*.json")
        
        print(f"  JPGファイル数: {img_count}")
        print(f"  PNGファイル数: {png_count}")
        print(f"  JSONファイル数: {json_count}")
        
        # サブディレクトリの確認
        subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        print(f"  サブディレクトリ: {', '.join(subdirs) if subdirs else 'なし'}")
    
    return True

def main():
    """メイン関数"""
    # コマンドライン引数からベースディレクトリを取得、指定がなければデフォルトを使用
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = "small_test_dataset"
    
    print_colored(f"small_test_datasetの構造チェックを開始します（パス: {base_dir}）", GREEN)
    
    # ディレクトリの存在確認
    if not os.path.exists(base_dir):
        print_colored(f"エラー: {base_dir} が存在しません", RED)
        return 1
    
    # 各種チェックの実行
    check_directory_structure(base_dir)
    check_refcoco_datasets(base_dir)
    check_other_datasets(base_dir)
    
    print_section("結論と推奨事項")
    print("1. データセットの問題点:")
    print("   - refer_segデータセットの画像IDとアノテーションファイルのIDが一致していない可能性があります")
    print("   - 元のデータセット（完全版）のアノテーションファイルがそのまま使用されています")
    
    print("\n2. 改善方法:")
    print("   - アノテーションファイルを実際のサンプル画像に合わせて再生成する")
    print("   - または、アノテーションファイルに対応する画像を追加する")
    
    print("\n3. 当面の対応策:")
    print("   - 学習コマンドから refer_seg を除外する（--dataset=\"sem_seg||vqa||reason_seg\"）")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 