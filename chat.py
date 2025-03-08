import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import re

from model.LISA import Llama32LISAForCausalLM
from model.llama3_2 import conversation as llama3_2_conversation
from model.llama3_2.constants import IMAGE_TOKEN, SEG_TOKEN
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

# Hugging Face認証を行う関数
def authenticate_huggingface():
    """Hugging Faceに認証を行います。環境変数を使用します。"""
    # 環境変数からトークンを取得
    hf_token = os.environ.get("HF_TOKEN")
    
    # Hugging Faceにログイン
    if hf_token:
        try:
            print("Hugging Faceに認証を行います...")
            login(token=hf_token)
            print("認証成功！")
        except Exception as e:
            print(f"Hugging Face認証エラー: {e}")
            print("認証なしで続行します。一部のモデルはアクセスできない可能性があります。")


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA chat")
    parser.add_argument("--version", default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default=None, type=str, help="Path to SAM ViT-H checkpoint"
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument(
        "--conv_type",
        default="llama3_2",
        type=str,
        choices=["llama3_2"],
    )
    parser.add_argument(
        "--sam_version",
        default=None,
        type=str,
        help="SAM model version, e.g., sam_vit_h_4b8939.pth"
    )
    parser.add_argument(
        "--model_type",
        default="llama32_lisa",
        type=str,
        choices=["llama32_lisa"],
        help="Model type: Llama3.2 Vision + SAM",
    )
    parser.add_argument(
        "--out_dim",
        default=256,
        type=int,
        help="Output dimension for mask embedding",
    )
    parser.add_argument(
        "--train_mask_decoder",
        action="store_true",
        default=False,
        help="Whether to train SAM's mask decoder",
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        default=True,
        help="Whether to ignore mismatched sizes when loading weights",
    )
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """SAM用の画像前処理"""
    # ピクセル値を正規化
    x = (x - pixel_mean) / pixel_std
    
    # サイズ調整
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    if padh > 0 or padw > 0:
        x = F.pad(x, (0, padw, 0, padh))
    return x


def get_prompt_template():
    """Llama 3.2のプロンプトテンプレートを取得"""
    return """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
あなたは役立つAIアシスタントです。画像セグメンテーションもできます。画像から特定のオブジェクトをセグメンテーションするように頼まれたら、次のような形式で回答してください：「<SEG> [セグメントする物体の説明]」

<|start_header_id|>user<|end_header_id|>
{input}

<|start_header_id|>assistant<|end_header_id|>
{response}
"""


def chatting(args, model, tokenizer, device, prompt_template, model_max_length, max_new_tokens, sep, stop_str):
    """チャットモードの実行関数"""
    print("チャットモードを開始します。'exit'と入力すると終了します。")
    
    # 会話履歴を初期化
    conversation = []
    
    while True:
        try:
            # ユーザー入力を取得
            user_input = input("\nユーザー: ")
            if user_input.lower() == "exit":
                break
            
            # 画像パスをチェック
            image_path = None
            for word in user_input.split():
                if os.path.exists(word) and word.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = word
                    user_input = user_input.replace(image_path, "").strip()
                    break
            
            # 会話履歴に追加
            conversation.append({"role": "user", "content": user_input})
            
            # 画像処理
            image_tensor = None
            if image_path:
                # 画像を読み込み
                image = Image.open(image_path).convert('RGB')
                
                # プロセッサで画像を処理
                inputs = tokenizer(
                    text=user_input,
                    images=image,
                    return_tensors="pt"
                )
                input_ids = inputs["input_ids"].to(device)
                
                # 画像テンソルも取得
                image_tensor = inputs["pixel_values"].to(device)
            else:
                # テキストのみの処理
                inputs = tokenizer.tokenizer(
                    user_input,
                    return_tensors="pt"
                )
                input_ids = inputs["input_ids"].to(device)
            
            # 生成パラメータを設定
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
            }
            
            # 画像があるかどうかで異なる生成処理
            if image_tensor is not None:
                # 画像と入力IDsを使って生成
                generate_kwargs["pixel_values"] = image_tensor
            
            # 生成を実行
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    **generate_kwargs
                )
            
            # 生成されたテキストをデコード
            generated_text = tokenizer.tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
            
            # レスポンス部分を抽出
            generated_response = generated_text.split(sep)[-1].strip()
            if stop_str in generated_response:
                generated_response = generated_response.split(stop_str)[0].strip()
            
            # SEGトークンがあるかチェック
            if "<SEG>" in generated_response:
                # SEGトークンのインデックスを検出
                seg_indices = []
                for i, token_id in enumerate(outputs.sequences[0]):
                    if token_id == tokenizer.tokenizer.convert_tokens_to_ids("<SEG>"):
                        seg_indices.append(i)
                
                if seg_indices and image_tensor is not None:
                    # 画像の元のサイズを取得
                    original_image = Image.open(image_path).convert('RGB')
                    original_size = original_image.size[::-1]  # (h, w)
                    
                    # SAM用に画像を前処理
                    processed_image = preprocess(image_tensor).to(device)
                    
                    # マスクを生成
                    masks = model.generate_masks(
                        images=processed_image,
                        input_ids=outputs.sequences,
                        seg_token_indices=seg_indices,
                        original_sizes=[original_size],
                    )
                    
                    # マスクを可視化
                    # 元の画像をOpenCV形式に変換
                    image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
                    
                    for i, mask in enumerate(masks):
                        # マスクをCPUに移動してNumPy配列に変換
                        mask_np = mask.cpu().numpy()
                        
                        # マスクを0-255の範囲にスケーリング
                        mask_np = (mask_np * 255).astype(np.uint8)
                        
                        # マスクを元の画像サイズにリサイズ
                        mask_np = cv2.resize(mask_np, (original_size[1], original_size[0]))
                        
                        # 画像とマスクのオーバーレイ
                        overlay = image_cv.copy()
                        colored_mask = np.zeros_like(image_cv)
                        colored_mask[mask_np > 127] = [0, 0, 255]  # 赤色でマスク領域を表示
                        
                        # マスクをブレンド
                        cv2.addWeighted(image_cv, 0.7, colored_mask, 0.3, 0, overlay)
                        
                        # 結果を保存
                        filename = f"mask_{i}_{Path(image_path).stem}.png"
                        save_path = os.path.join(args.vis_save_path, filename)
                        cv2.imwrite(save_path, overlay)
                        print(f"マスク画像を保存しました: {save_path}")
            
            # 応答を表示
            print(f"\nアシスタント: {generated_response}")
            
            # 会話履歴に追加
            conversation.append({"role": "assistant", "content": generated_response})
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            import traceback
            traceback.print_exc()


def main(args):
    # Hugging Face認証
    authenticate_huggingface()
    
    # 出力ディレクトリの作成
    if not os.path.exists(args.vis_save_path):
        os.makedirs(args.vis_save_path)

    # モデルの読み込み
    print(f"Loading Llama3.2 Vision + SAM model: {args.version}")
    
    # トークナイザーをロード
    tokenizer = AutoProcessor.from_pretrained(args.version)
    
    # <SEG>トークンをトークナイザーに追加
    if "<SEG>" not in tokenizer.tokenizer.get_vocab():
        print("Adding <SEG> token to tokenizer vocabulary")
        tokenizer.tokenizer.add_special_tokens({"additional_special_tokens": ["<SEG>"]})
    
    # from_vision_modelメソッドを使用してモデルとトークナイザーを同時に取得
    model, tokenizer = Llama32LISAForCausalLM.from_vision_model(
        vision_model_id=args.version,
        vision_pretrained=args.sam_version,
        train_mask_decoder=args.train_mask_decoder,
        tokenizer=tokenizer,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes
    )
    
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # チャットループの設定
    prompt_template = get_prompt_template()
    model_max_length = tokenizer.tokenizer.model_max_length
    max_new_tokens = args.max_new_tokens
    sep = "[/INST]"
    stop_str = "</s>"
    
    # チャットモードの起動
    chatting(args, model, tokenizer, device, prompt_template, model_max_length, max_new_tokens, sep, stop_str)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
