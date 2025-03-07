#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Llama3.2 Vision + SAMをテストするための簡単なスクリプト
"""

import os
import argparse
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

from model.llama3_2.hf_auth import login_huggingface
from model.llama3_2.utils import (
    preprocess_image_for_llama32, 
    preprocess_image_for_sam,
    format_prompt,
    SEG_TOKEN
)
from model.segment_anything import build_sam_vit_h
from model.segment_anything.utils.transforms import ResizeLongestSide

def parse_args():
    parser = argparse.ArgumentParser(description="Llama3.2 Vision + SAMテスト")
    parser.add_argument("--image_path", type=str, required=True, help="テスト画像へのパス")
    parser.add_argument("--sam_checkpoint", type=str, default="./checkpoints/sam_vit_h_4b8939.pth", 
                        help="SAMチェックポイントへのパス")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct",
                       help="Llama3.2のモデルID")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="bf16",
                       help="モデルの精度")
    parser.add_argument("--output_dir", type=str, default="./test_output",
                       help="出力ディレクトリ")
    parser.add_argument("--query", type=str, default="この画像について説明し、主要な物体を[SEG]でセグメンテーションしてください。",
                       help="モデルへの問い合わせ")
    parser.add_argument("--hf_token", type=str, default=None,
                       help="Hugging Faceトークン")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Hugging Faceにログイン
    if args.hf_token:
        login_huggingface(args.hf_token)
    
    print(f"モデル {args.model_id} をロード中...")
    
    # モデルとプロセッサの準備
    torch_dtype = torch.bfloat16 if args.precision == "bf16" else \
                 torch.float16 if args.precision == "fp16" else torch.float32
    
    # Llama3.2 Visionモデルをロード
    processor = AutoProcessor.from_pretrained(args.model_id)
    llama_model = MllamaForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map="auto"
    )
    
    # SAMモデルをロード
    print(f"SAMモデルをロード中... {args.sam_checkpoint}")
    sam_model = build_sam_vit_h(args.sam_checkpoint)
    sam_model.to(device=llama_model.device, dtype=torch_dtype)
    sam_model.eval()
    
    # 画像の読み込み
    print(f"画像を読み込み中... {args.image_path}")
    image_np = cv2.imread(args.image_path)
    if image_np is None:
        print(f"画像が読み込めませんでした: {args.image_path}")
        return
    
    # RGB形式に変換（OpenCVはBGR形式で読み込む）
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    # SAM用の画像前処理
    transform = ResizeLongestSide(1024)
    sam_image_tensor, resize_size, original_size = preprocess_image_for_sam(image_np, transform)
    
    # SAM画像をGPUに転送
    sam_image_tensor = sam_image_tensor.to(device=llama_model.device, dtype=torch_dtype)
    
    # 画像をLlama3.2のプロセッサで処理
    query = args.query
    print(f"問い合わせ: {query}")
    
    # プロセッサで画像とテキストを処理
    formatted_prompt = format_prompt(query)
    print(f"フォーマット済みプロンプト: {formatted_prompt}")
    
    inputs = processor(
        images=Image.fromarray(image_np),
        text=formatted_prompt,
        return_tensors="pt"
    ).to(llama_model.device)
    
    # 生成
    print("生成中...")
    with torch.no_grad():
        # テキスト生成
        output = llama_model.generate(
            **inputs,
            max_new_tokens=512,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        
        # 隠れ状態から<SEG>トークンの位置を見つける
        # トークンを復号化して<SEG>の位置を探す
        generated_text = processor.decode(output.sequences[0], skip_special_tokens=False)
        print(f"生成テキスト: {generated_text}")
        
        # <SEG>トークンの位置を特定
        if SEG_TOKEN in generated_text:
            print(f"{SEG_TOKEN}トークンが見つかりました - SAMで処理します")
            
            # 最後の隠れ状態を取得
            last_hidden_states = output.hidden_states[-1]
            
            # <SEG>トークンに対応する隠れ状態を取得
            # ここでは簡略化のため、最後の隠れ状態を使用
            # 実際には<SEG>トークンの正確な位置に対応する隠れ状態を使用する必要があります
            seg_embedding = last_hidden_states[0, -1].unsqueeze(0)
            
            # SAMのプロンプトエンコーダで処理
            # プロンプトエンコーダの入力次元に合わせるために簡単な投影層を追加
            # 実際には事前に訓練された投影層を使用するべきです
            projection = torch.nn.Linear(seg_embedding.shape[-1], 256).to(llama_model.device, dtype=torch_dtype)
            seg_embedding_projected = projection(seg_embedding)
            
            # SAMのプロンプトエンコーダを使用
            with torch.no_grad():
                image_embedding = sam_model.image_encoder(sam_image_tensor)
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=seg_embedding_projected.unsqueeze(1)
                )
                
                # SAMのマスクデコーダを使用してマスクを生成
                low_res_masks, _ = sam_model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )
                
                # マスクを後処理
                masks = sam_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_size,
                    original_size=original_size
                )
                
                # マスクをバイナリ形式に変換
                mask = masks[0, 0].cpu().numpy() > 0
                
                # 結果を保存
                base_name = os.path.basename(args.image_path).split('.')[0]
                
                # マスク画像を保存
                mask_path = os.path.join(args.output_dir, f"{base_name}_mask.png")
                cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)
                print(f"マスクを保存しました: {mask_path}")
                
                # マスク付き画像を保存
                masked_img = image_np.copy()
                masked_img[mask] = (
                    image_np * 0.5 + 
                    mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
                )[mask]
                
                masked_img_path = os.path.join(args.output_dir, f"{base_name}_masked.png")
                cv2.imwrite(masked_img_path, cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))
                print(f"マスク付き画像を保存しました: {masked_img_path}")
        else:
            print(f"{SEG_TOKEN}トークンが見つかりませんでした - マスクは生成されません")
            
        # 生成テキストを保存
        text_path = os.path.join(args.output_dir, f"{os.path.basename(args.image_path).split('.')[0]}_response.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(generated_text)
        print(f"生成テキストを保存しました: {text_path}")

if __name__ == "__main__":
    main() 