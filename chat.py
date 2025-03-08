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
    parser = argparse.ArgumentParser(description='LISA Chat')
    
    # モデルと設定
    parser.add_argument(
        "--version",
        default="meta-llama/Llama-3.2-11B-Vision-Instruct",
        help="LISA model version",
    )
    parser.add_argument(
        "--sam_version",
        type=str,
        default="./checkpoints/sam_vit_h_4b8939.pth",
        help="SAM model version",
    )
    parser.add_argument(
        "--vision_tower",
        type=str,
        default=None,
        help="Vision tower model name or path",
    )
    parser.add_argument(
        "--out_dim", 
        type=int, 
        default=256,
        help="Output dimension"
    )
    parser.add_argument(
        '--precision',
        type=str,
        default='bf16',
        choices=['fp32', 'bf16', 'fp16'],
        help='Precision for model weights'
    )
    parser.add_argument(
        "--vis_save_path",
        type=str,
        default="./vis_output/",
        help="Path to save visualization results",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=1024,
        help="image size for SAM model input",
    )
    
    # 生成設定
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling, 0 means greedy decoding",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling probability",
    )
    
    # メモリ最適化オプション
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        default=False,
        help="Load model in 8-bit precision",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=False,
        help="Load model in 4-bit precision",
    )
    parser.add_argument(
        "--low_memory",
        action="store_true",
        default=False,
        help="Enable low memory mode for smaller images and more memory efficiency",
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
    parser.add_argument(
        "--very_low_memory",
        action="store_true",
        default=False,
        help="Enable very low memory mode for smaller images and more memory efficiency",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        default=False,
        help="Enable mixed precision inference",
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
    
    # デバイスマップの確認 - モデルがautoマッピングを使用しているかどうか
    device_map = getattr(model, "hf_device_map", None)
    is_auto_device_map = device_map == "auto" or isinstance(device_map, dict)
    
    # 最大画像サイズの設定（メモリ使用量削減のため）
    if args.very_low_memory:
        max_img_size = 384  # 超低メモリモードでは384x384に制限
    elif args.low_memory:
        max_img_size = 512  # 低メモリモードでは512x512に制限
    else:
        max_img_size = 768  # 通常モードでも768x768に制限
    
    print(f"画像処理の最大サイズ: {max_img_size}x{max_img_size}")
    
    # 常にGCを有効化してメモリリークを防止
    import gc
    gc.enable()
    
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
            
            # 処理開始前にメモリクリーンアップ
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # 画像処理
            image_tensor = None
            if image_path:
                # 画像を読み込み - メモリ効率のため小さめのサイズで読み込む
                image = Image.open(image_path).convert('RGB')
                
                # 画像サイズを制限（元のアスペクト比を維持）
                w, h = image.size
                if max(w, h) > max_img_size:
                    ratio = max_img_size / max(w, h)
                    new_size = (int(w * ratio), int(h * ratio))
                    print(f"画像リサイズ: {w}x{h} -> {new_size[0]}x{new_size[1]}")
                    image = image.resize(new_size, Image.LANCZOS)
                
                # 推論中に不要なメモリを解放するためno_gradを使用
                with torch.no_grad():
                    # プロセッサで画像を処理
                    inputs = tokenizer(
                        text=user_input,
                        images=image,
                        return_tensors="pt"
                    )
                    
                    # 入力をデバイスに送る
                    if not is_auto_device_map:
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    input_ids = inputs["input_ids"]
                    image_tensor = inputs["pixel_values"]
            else:
                # テキストのみの処理
                with torch.no_grad():
                    inputs = tokenizer.tokenizer(
                        user_input,
                        return_tensors="pt"
                    )
                    if not is_auto_device_map:
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                    input_ids = inputs["input_ids"]
            
            # 生成パラメータを設定
            generate_kwargs = {
                "max_new_tokens": min(max_new_tokens, 256 if args.very_low_memory else 512),  # 超低メモリモードではトークン数も制限
                "do_sample": True if args.temperature > 0 else False,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
            
            # 画像があるかどうかで異なる生成処理
            if image_tensor is not None:
                # 画像と入力IDsを使って生成
                generate_kwargs["pixel_values"] = image_tensor
            
            # 生成を実行 - torch.no_gradでメモリ使用量を削減
            with torch.no_grad():
                try:
                    # 生成中にGPUメモリ不足になる場合は、より小さなメモリフットプリントでの生成を試みる
                    outputs = model.generate(
                        input_ids=input_ids,
                        return_dict_in_generate=True,
                        output_hidden_states=True,
                        **generate_kwargs
                    )
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print("警告: GPUメモリ不足のため、より小さなトークン数で再試行します")
                        # トークン数を減らして再試行
                        generate_kwargs["max_new_tokens"] = 128
                        # さらにメモリをクリーンアップ
                        del input_ids
                        if image_tensor is not None:
                            del image_tensor
                        torch.cuda.empty_cache()
                        gc.collect()
                        # 再度入力を処理
                        if image_path:
                            # 画像をさらに小さいサイズに
                            image = Image.open(image_path).convert('RGB')
                            max_retry_size = 256  # 再試行時はさらに小さく
                            w, h = image.size
                            ratio = max_retry_size / max(w, h)
                            new_size = (int(w * ratio), int(h * ratio))
                            print(f"再試行: 画像を縮小 {new_size[0]}x{new_size[1]}")
                            image = image.resize(new_size, Image.LANCZOS)
                            
                            inputs = tokenizer(
                                text=user_input,
                                images=image,
                                return_tensors="pt"
                            )
                        else:
                            inputs = tokenizer.tokenizer(
                                user_input,
                                return_tensors="pt"
                            )
                        
                        if not is_auto_device_map:
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        input_ids = inputs["input_ids"]
                        if "pixel_values" in inputs:
                            generate_kwargs["pixel_values"] = inputs["pixel_values"]
                        
                        # 再度生成
                        outputs = model.generate(
                            input_ids=input_ids,
                            return_dict_in_generate=True,
                            output_hidden_states=False,  # メモリ削減のため
                            **generate_kwargs
                        )
                    else:
                        raise
            
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
                
                if seg_indices and image_path:
                    # 画像の元のサイズを取得
                    original_image = Image.open(image_path).convert('RGB')
                    original_size = original_image.size[::-1]  # (h, w)
                    
                    # マスク生成に必要なイメージデータを準備
                    # 元の画像をOpenCV形式に変換（マスク可視化用）
                    image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
                    
                    # SAM用に画像を前処理
                    # 低メモリモードの場合、より小さいサイズでSAM処理
                    sam_image_size = 256 if args.very_low_memory else (384 if args.low_memory else 512)
                    
                    # メモリ効率のため、ここでプロセッサで処理した画像を使用
                    with torch.no_grad():
                        # SAM用の画像を前処理
                        processed_image = preprocess(image_tensor, img_size=sam_image_size)
                        if not is_auto_device_map:
                            processed_image = processed_image.to(device)
                        
                        # マスクを生成
                        masks = model.generate_masks(
                            images=processed_image,
                            input_ids=outputs.sequences,
                            seg_token_indices=seg_indices,
                            original_sizes=[original_size],
                        )
                    
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
                        filename = f"{Path(image_path).stem}_masked_img_{i}.jpg"
                        save_path = os.path.join(args.vis_save_path, filename)
                        cv2.imwrite(save_path, overlay)
                        print(f"マスク画像を保存しました: {save_path}")
                        
                        # メモリ解放（大きなNumPy配列）
                        del mask_np, overlay, colored_mask
                    
                    # メモリ解放
                    del masks, processed_image, image_cv
                
                # 明示的に不要なテンソルを解放
                if 'image_tensor' in locals() and image_tensor is not None:
                    del image_tensor
                
                if 'inputs' in locals():
                    del inputs
                
                if 'input_ids' in locals():
                    del input_ids
                
                if 'outputs' in locals():
                    del outputs
                
                # ガベージコレクションを実行してメモリを確実に解放
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"GPUメモリ使用状況: {torch.cuda.memory_allocated() / 1024**2:.2f} MB / {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
            
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
            # エラー発生時もメモリをクリーンアップ
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main(args):
    # Hugging Face認証
    authenticate_huggingface()
    
    # 出力ディレクトリの作成
    if not os.path.exists(args.vis_save_path):
        os.makedirs(args.vis_save_path)

    # モデルの読み込み
    print(f"Loading Llama3.2 Vision + SAM model: {args.version}")
    
    # モデル精度の設定
    if args.precision == "fp32":
        dtype = torch.float32
    elif args.precision == "bf16":
        dtype = torch.bfloat16
    elif args.precision == "fp16":
        dtype = torch.float16
    else:
        raise ValueError(f"Unsupported precision: {args.precision}")
    
    # 4bit/8bit量子化をスキップするためのチェック
    if args.load_in_4bit or args.load_in_8bit:
        print("警告: bitsandbytesライブラリのセットアップに問題があるため、4bit/8bit量子化をスキップします")
        print("代わりにGPUメモリ最適化を実施します")
        args.load_in_4bit = False
        args.load_in_8bit = False
        args.very_low_memory = True
    
    # トークナイザーをロード
    tokenizer = AutoProcessor.from_pretrained(args.version)
    
    # <SEG>トークンをトークナイザーに追加
    if "<SEG>" not in tokenizer.tokenizer.get_vocab():
        print("Adding <SEG> token to tokenizer vocabulary")
        tokenizer.tokenizer.add_special_tokens({"additional_special_tokens": ["<SEG>"]})
    
    # デバイスマップを設定 - メモリ使用量を削減するための特別設定
    device_map = None
    
    # 超低メモリモードの場合、一部のレイヤーをCPUにオフロード
    if args.very_low_memory:
        print("超低メモリモードを有効化します - 一部のレイヤーをCPUにオフロード")
        device_map = {
            "model.embed_tokens": 0,
            "model.norm": 0,
            "model.layers.0": 0, 
            "model.layers.1": 0,
            "model.layers.2": 0,
            "model.layers.3": 0,
            "model.layers.4": 0,
            "model.layers.5": 0,
            "model.layers.6": 0,
            "model.layers.7": 0,
            "model.layers.8": 0,
            "model.layers.9": 0,
            "model.layers.10": 0,
            "model.layers.11": 0,
            "model.layers.12": 0,
            "model.layers.13": 0,
            "model.layers.14": 0,
            "model.layers.15": 0,
            "model.layers.16": "cpu",
            "model.layers.17": "cpu",
            "model.layers.18": "cpu",
            "model.layers.19": "cpu",
            "model.layers.20": "cpu",
            "model.layers.21": "cpu",
            "model.layers.22": "cpu",
            "model.layers.23": "cpu",
            "model.layers.24": "cpu",
            "model.layers.25": "cpu",
            "model.layers.26": "cpu",
            "model.layers.27": "cpu",
            "model.layers.28": "cpu",
            "model.layers.29": "cpu",
            "model.layers.30": "cpu",
            "model.layers.31": "cpu",
            "lm_head": 0,
            "vision_tower": 0,
        }
    elif args.low_memory:
        print("低メモリモードを有効化します")
        # auto device mapを使用
        device_map = "auto"
    
    # from_vision_modelメソッドを使用してモデルとトークナイザーを同時に取得
    # メモリ効率を高めるためにdevice_mapを使用
    model, tokenizer = Llama32LISAForCausalLM.from_vision_model(
        vision_model_id=args.version,
        vision_pretrained=args.sam_version,
        train_mask_decoder=args.train_mask_decoder,
        tokenizer=tokenizer,
        torch_dtype=dtype,
        device_map=device_map,  # 自動的にGPU/CPU間でレイヤーを配置
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        load_in_8bit=False,  # BNBエラーのため無効化
        load_in_4bit=False   # BNBエラーのため無効化
    )
    
    # モデルをデバイスに配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデルが混合精度を使用するように設定（オプション）
    if args.mixed_precision and torch.cuda.is_available():
        print("混合精度推論を有効化します")
        amp_config = {"enabled": True, "dtype": dtype}
    else:
        amp_config = {"enabled": False}
    
    # チャットループの設定
    prompt_template = get_prompt_template()
    model_max_length = tokenizer.tokenizer.model_max_length
    max_new_tokens = args.max_new_tokens
    sep = "[/INST]"
    stop_str = "</s>"
    
    # メモリの状態を報告
    if torch.cuda.is_available():
        print(f"GPUメモリ使用状況: {torch.cuda.memory_allocated() / 1024**2:.2f} MB / {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    
    # チャットモードの起動
    chatting(args, model, tokenizer, device, prompt_template, model_max_length, max_new_tokens, sep, stop_str)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
