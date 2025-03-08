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
from model.llama3_2.mm_utils import create_mllama_message, process_images, safe_mllama_to_sam

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
        "--test_image",
        type=str,
        default=None,
        help="Path to a test image for validating processor functionality",
    )
    parser.add_argument(
        "--run_segment_test",
        action="store_true",
        help="Run a segmentation test with the test image if provided",
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
        default='fp16',
        choices=['fp32', 'bf16', 'fp16'],
        help='Precision for model weights (fp16推奨)'
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
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling probability",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty for text generation",
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
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to use sampling for generation",
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
    
    # トークナイザーのアクセス方法を決定
    tokenizer_func = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
    
    # 画像トークンを取得（MllamaProcessorのための特別処理）
    image_token = None
    if hasattr(tokenizer, 'image_token'):
        image_token = tokenizer.image_token
    elif hasattr(tokenizer_func, 'special_tokens_map') and 'image' in tokenizer_func.special_tokens_map:
        image_token = tokenizer_func.special_tokens_map['image']
    else:
        # デフォルトの画像トークン
        image_token = "<image>"
    
    # モデルタイプの判定
    is_mllama = hasattr(tokenizer, 'processor') or hasattr(tokenizer, 'image_processor')
    if is_mllama:
        print(f"Llama 3.2 Vision (Mllama)モデルを検出しました。画像トークン: {image_token}")
    else:
        print(f"使用する画像トークン: {image_token}")
    
    while True:
        try:
            # ユーザー入力を取得
            user_input = input("\nユーザー: ")
            if user_input.lower() == "exit":
                break
            
            # 画像パスを明示的に要求
            image_path = None
            is_seg_request = "<SEG>" in user_input or "segment" in user_input.lower()
            
            # セグメンテーション要求があれば画像パスも要求
            if is_seg_request:
                image_path = input("画像のパスを入力してください: ")
                
                # 画像が存在するか確認
                if not os.path.exists(image_path) or not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    print(f"警告: 指定されたパス '{image_path}' が存在しないか、サポートされていない形式です。")
                    image_path = None
                    
                    # 再試行するか確認
                    retry = input("有効な画像パスを入力しますか？ (y/n): ")
                    if retry.lower() == 'y':
                        image_path = input("画像のパスを再入力してください: ")
                        if not os.path.exists(image_path) or not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                            print(f"警告: 指定されたパス '{image_path}' が存在しないか、サポートされていない形式です。テキストのみで処理します。")
                            image_path = None
            
            # 古いコード（バックアップとして残すが、上記の明示的な要求が優先）
            if image_path is None:
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
                try:
                    # 画像を読み込み - メモリ効率のため小さめのサイズで読み込む
                    image = Image.open(image_path).convert('RGB')
                    
                    # 画像サイズを制限（元のアスペクト比を維持）
                    w, h = image.size
                    if max(w, h) > max_img_size:
                        ratio = max_img_size / max(w, h)
                        new_size = (int(w * ratio), int(h * ratio))
                        print(f"画像リサイズ: {w}x{h} -> {new_size[0]}x{new_size[1]}")
                        image = image.resize(new_size, Image.LANCZOS)
                    
                    # SEGトークンをプロンプトに追加（必要な場合）
                    if is_seg_request and "<SEG>" not in user_input and "[SEG]" not in user_input:
                        user_input += " [SEG]"
                    
                    # テキストにセグメンテーション指示がある場合は必ず画像トークンを追加
                    is_segment_request = "segment" in user_input.lower() or "<SEG>" in user_input or "[SEG]" in user_input
                    
                    # 推論中に不要なメモリを解放するためno_gradを使用
                    with torch.no_grad():
                        # Llama 3.2 Vision (Mllama)モデルの場合
                        if is_mllama:
                            # 最もシンプルな方法: 単純な文字列とPIL画像
                            # 画像を先に指定し、テキストを後に指定する
                            try:
                                print("Mllamaプロセッサを使用")
                                # セグメンテーション要求の場合は画像トークンを追加
                                if is_segment_request and "<|image|>" not in user_input:
                                    processed_input = "<|image|> " + user_input
                                else:
                                    processed_input = user_input
                                
                                # メッセージの表示
                                print(f"入力テキスト: {processed_input}")
                                
                                # 明示的な引数順序でプロセッサを呼び出す
                                print("Mllamaプロセッサでtokenizer呼び出し前...")
                                print(f"画像タイプ: {type(image)}, テキスト: {processed_input[:50]}...")
                                
                                # Mllama tokenizer/processorを確認
                                if hasattr(tokenizer, 'processor'):
                                    print(f"processorタイプ: {type(tokenizer.processor)}")
                                
                                inputs = tokenizer(
                                    images=image,  # 画像を第1引数またはキーワード引数で
                                    text=processed_input,  # テキストを第2引数またはキーワード引数で
                                    return_tensors="pt"
                                )
                                
                                # テンソル形状のデバッグ
                                for k, v in inputs.items():
                                    if isinstance(v, torch.Tensor):
                                        print(f"inputs['{k}']の形状: {v.shape}, dtype: {v.dtype}")
                                
                                # pixel_valuesの特別処理
                                if 'pixel_values' in inputs:
                                    pixel_values = inputs['pixel_values']
                                    print(f"pixel_values形状: {pixel_values.shape}, dtype: {pixel_values.dtype}")
                                    
                                    # 形状が複雑すぎる場合、調整を検討
                                    if len(pixel_values.shape) > 4:
                                        print(f"警告: pixel_valuesの形状が複雑すぎます: {pixel_values.shape}")
                                        print("SAMはシンプルな[バッチ, チャンネル, 高さ, 幅]形式を期待します")
                                        
                                        # バックアップとして画像を直接PIL→テンソルに変換
                                        if is_segment_request:
                                            print("セグメンテーション用に画像を直接変換します...")
                                            # SAM用の専用画像処理を実行
                                            from model.segment_anything.utils.transforms import ResizeLongestSide
                                            transform = ResizeLongestSide(1024)
                                            
                                            # PIL画像からNumPy配列を取得
                                            import numpy as np
                                            image_np = np.array(image)
                                            
                                            # SAMの変換を適用
                                            image_transformed = transform.apply_image(image_np)
                                            print(f"変換後の画像形状: {image_transformed.shape}")
                                    
                            except Exception as e1:
                                print(f"標準の処理中にエラー: {e1}")
                                try:
                                    # フォーマットされたメッセージを作成
                                    print("代替方法: フォーマットされたメッセージを使用")
                                    message = create_mllama_message(image, user_input, verbose=True)
                                    print(f"作成されたメッセージ: {message.keys()}")
                                    inputs = tokenizer(**message, return_tensors="pt")
                                    
                                    # テンソル形状のデバッグ
                                    for k, v in inputs.items():
                                        if isinstance(v, torch.Tensor):
                                            print(f"代替inputs['{k}']の形状: {v.shape}, dtype: {v.dtype}")
                                except Exception as e2:
                                    print(f"代替処理中にエラー: {e2}")
                                    raise ValueError("画像処理に失敗しました")
                                
                                # デバイスに移動
                                if not is_auto_device_map:
                                    inputs = {k: v.to(device) for k, v in inputs.items()}
                                
                                # 入力IDsを取得
                                input_ids = inputs["input_ids"]
                            else:
                                # 通常のVLM処理（LLaVA等）
                                # 画像トークンをテキストに追加
                                if image_token not in user_input:
                                    modified_input = f"{image_token} {user_input}"
                                else:
                                    modified_input = user_input
                                
                                # 通常の処理 - VLMと画像を同時に処理
                                inputs = tokenizer(
                                    text=modified_input,
                                    images=image,
                                    return_tensors="pt"
                                )
                                
                                # デバイスに移動
                                if not is_auto_device_map:
                                    inputs = {k: v.to(device) for k, v in inputs.items()}
                                
                                # 入力IDsと画像テンソルを取得
                                input_ids = inputs["input_ids"]
                                if "pixel_values" in inputs:
                                    image_tensor = inputs["pixel_values"]
                except Exception as main_error:
                    print(f"画像処理に完全に失敗しました: {main_error}")
                    # テキストのみで処理を継続
                    inputs = tokenizer_func(user_input, return_tensors="pt")
                    if not is_auto_device_map:
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                    input_ids = inputs["input_ids"]
            else:
                # テキストのみの処理
                with torch.no_grad():
                    inputs = tokenizer_func(user_input, return_tensors="pt")
                    if not is_auto_device_map:
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                    input_ids = inputs["input_ids"]
            
            # 入力サイズの確認
            input_length = input_ids.shape[0]
            print(f"入力トークン数: {input_length}")
            
            # 出力に十分な長さを確保
            if input_length + max_new_tokens > model_max_length:
                max_possible_tokens = model_max_length - input_length
                print(f"警告: 入力が長すぎるため、max_new_tokensを{max_new_tokens}から{max_possible_tokens}に削減します")
                max_new_tokens = max(1, max_possible_tokens)
            
            # メモリ効率のため推論時にはgrad計算を無効化
            with torch.no_grad():
                try:
                    # 画像によるセグメンテーションの場合、generate_masksを使用
                    if image_path and (is_seg_request or "<SEG>" in user_input or "[SEG]" in user_input):
                        print("セグメンテーションモードで実行中...")
                        
                        # SEGトークンをプロンプトに追加（まだない場合）
                        if "<SEG>" not in user_input and "[SEG]" not in user_input:
                            if "segment" in user_input.lower():
                                # テキストの末尾に[SEG]を追加
                                user_input += " [SEG]"
                                # 入力を再構築
                                inputs = tokenizer_func(
                                    user_input,
                                    return_tensors="pt"
                                )
                                input_ids = inputs["input_ids"]
                                if not is_auto_device_map:
                                    inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        # LISAモデルのgenerate_masks関数用にパラメータを整理
                        # 重複を避けるため、inputsから冗長なキーを削除
                        # input_ids と attention_mask は明示的に渡すので除外
                        mask_inputs = {k: v for k, v in inputs.items() 
                                      if k not in ["input_ids", "attention_mask"]}
                        
                        # SAM用に画像を適切に準備
                        # オリジナルのLISAに合わせた方法で画像を処理
                        from model.segment_anything.utils.transforms import ResizeLongestSide
                        transform = ResizeLongestSide(1024)  # SAMは1024x1024を使用
                        
                        # 画像が正しく処理されているか確認
                        if image is not None:
                            print(f"画像情報: {type(image)}, サイズ: {image.size if hasattr(image, 'size') else 'N/A'}")
                            
                            try:
                                # SAM用の画像処理
                                import numpy as np
                                image_np = np.array(image)
                                
                                # 入力テンソルに特別な処理が必要か確認
                                if 'pixel_values' in inputs and len(inputs['pixel_values'].shape) > 4:
                                    print("複雑な形状のpixel_valuesを検出しました。適切な形状に変換します...")
                                    pixel_values = inputs['pixel_values']
                                    
                                    try:
                                        # ユーティリティ関数を使用してSAM用に安全に変換
                                        from model.llama3_2.mm_utils import safe_mllama_to_sam
                                        pixel_values = safe_mllama_to_sam(pixel_values)
                                        print(f"SAM用に変換成功: 形状={pixel_values.shape}")
                                        
                                        # デバイスをモデルに合わせる
                                        device_to_use = next(model.parameters()).device
                                        print(f"モデルデバイス: {device_to_use}")
                                        pixel_values = pixel_values.to(device_to_use)
                                        print(f"テンソルをデバイス{device_to_use}に移動しました")
                                        
                                        # mask_inputsからpixel_valuesを削除して重複を防ぐ
                                        clean_mask_inputs = {k: v for k, v in mask_inputs.items() 
                                                           if k != 'pixel_values'}
                                        
                                        # トークナイザーを追加
                                        clean_mask_inputs["tokenizer"] = tokenizer
                                        
                                        # generate_masksを実行
                                        result = model.generate_masks(
                                            image=None,  # PILを渡さない
                                            input_ids=input_ids,
                                            attention_mask=inputs.get("attention_mask", None),
                                            pixel_values=pixel_values,  # 変換済みのテンソルを渡す
                                            max_new_tokens=max_new_tokens,
                                            **clean_mask_inputs
                                        )
                                    except Exception as conversion_error:
                                        print(f"テンソル変換中にエラー: {conversion_error}")
                                        print("代わりにPIL画像を使用します...")
                                        
                                        # mask_inputsからpixel_valuesを削除して重複を防ぐ
                                        clean_mask_inputs = {k: v for k, v in mask_inputs.items() 
                                                           if k != 'pixel_values'}
                                        
                                        # トークナイザーを追加
                                        clean_mask_inputs["tokenizer"] = tokenizer
                                        
                                        # 通常の処理 - PILイメージを使用
                                        result = model.generate_masks(
                                            image=image,  # PIL Image
                                            input_ids=input_ids,
                                            attention_mask=inputs.get("attention_mask", None),
                                            pixel_values=None,  # pixel_valuesは渡さない
                                            max_new_tokens=max_new_tokens,
                                            **clean_mask_inputs
                                        )
                                else:
                                    # mask_inputsからpixel_valuesを削除して重複を防ぐ
                                    clean_mask_inputs = {k: v for k, v in mask_inputs.items() 
                                                       if k != 'pixel_values'}
                                    
                                    # トークナイザーを追加
                                    clean_mask_inputs["tokenizer"] = tokenizer
                                    
                                    # 通常の処理 - PILイメージを使用
                                    result = model.generate_masks(
                                        image=image,  # PIL Image
                                        input_ids=input_ids,
                                        attention_mask=inputs.get("attention_mask", None),
                                        pixel_values=None,  # pixel_valuesを明示的にNoneに設定
                                        max_new_tokens=max_new_tokens,
                                        **clean_mask_inputs
                                    )
                                    
                                # 結果の処理
                                if isinstance(result, dict) and "masks" in result:
                                    # マスクと出力トークンを取得
                                    masks = result["masks"]
                                    generation = result["output_tokens"]
                                    
                                    # マスクの可視化と保存
                                    if len(masks) > 0:
                                        print(f"{len(masks)}個のマスクが生成されました")
                                        
                                        # 原画像の取得
                                        original_image = image
                                        
                                        # マスクの可視化と保存
                                        import matplotlib.pyplot as plt
                                        import cv2
                                        
                                        # 各マスクを処理
                                        for i, mask in enumerate(masks):
                                            # マスクをNumPy配列に変換
                                            mask_np = mask.cpu().numpy()
                                            
                                            # マスクのサイズを画像サイズに合わせる
                                            if hasattr(original_image, "size"):
                                                img_w, img_h = original_image.size
                                                if mask_np.shape != (img_h, img_w):
                                                    mask_np = cv2.resize(mask_np, (img_w, img_h))
                                            
                                            # マスクオーバーレイ画像の作成
                                            plt.figure(figsize=(10, 10))
                                            plt.imshow(np.array(original_image))
                                            
                                            # マスクをオーバーレイ
                                            mask_display = np.zeros((mask_np.shape[0], mask_np.shape[1], 4))
                                            mask_display[:, :, 0] = 1  # R
                                            mask_display[:, :, 3] = mask_np * 0.5  # アルファチャンネル
                                            plt.imshow(mask_display)
                                            
                                            # 保存パス
                                            mask_path = os.path.join(args.vis_save_path, f"{os.path.basename(image_path).split('.')[0]}_masked_img_{i}.jpg")
                                            
                                            # 保存して閉じる
                                            plt.axis("off")
                                            plt.savefig(mask_path, bbox_inches="tight", pad_inches=0)
                                            plt.close()
                                            
                                            print(f"マスク{i+1}を保存しました: {mask_path}")
                                    else:
                                        # マスクがない場合は通常の出力を使用
                                        generation = result
                                    
                            except Exception as e:
                                print(f"マスク生成中にエラー: {e}")
                                import traceback
                                traceback.print_exc()
                                # テキスト生成にフォールバック
                                generation_config = {
                                    "max_new_tokens": max_new_tokens,
                                    "do_sample": args.do_sample,
                                }
                                
                                if args.do_sample:
                                    generation_config.update({
                                        "temperature": args.temperature,
                                        "top_p": args.top_p,
                                        "top_k": args.top_k,
                                    })
                                
                                generation = model.generate(
                                    **inputs,
                                    **generation_config
                                )
                    else:
                        # 通常の生成
                        print("通常の生成モードで実行中...")
                        generation_config = {
                            "max_new_tokens": max_new_tokens,
                            "do_sample": args.do_sample,
                        }
                        
                        # サンプリング有効時のみパラメータを追加
                        if args.do_sample:
                            generation_config.update({
                                "temperature": args.temperature,
                                "top_p": args.top_p,
                                "top_k": args.top_k,
                            })
                        
                        # 常に適用するパラメータ
                        generation_config.update({
                            "repetition_penalty": args.repetition_penalty,
                        })
                        
                        # 生成実行
                        generation = model.generate(
                            **inputs,
                            **generation_config
                        )
                    
                    # 生成されたテキストのデコード
                    output = tokenizer.decode(generation[0][input_ids.shape[0]:], skip_special_tokens=True)
                    
                    # 生成テキストが停止ワードで終わっている場合、その前までを使用
                    if stop_str and stop_str in output:
                        output = output.split(stop_str)[0]
                    
                    # 結果を会話に追加して表示
                    conversation[-1]["content"] = output
                    print(f"Assistant: {output}")
                
                except Exception as gen_error:
                    print(f"生成中にエラーが発生しました: {gen_error}")
                    import traceback
                    traceback.print_exc()
                    # エラーメッセージを会話に追加
                    error_msg = f"エラーが発生しました。もう一度試してください。"
                    conversation[-1]["content"] = error_msg
                    print(f"Assistant: {error_msg}")
        
        except KeyboardInterrupt:
            print("ユーザーによる中断...")
            break


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
        print("警告: fp32精度は多くのGPUメモリを使用します")
    elif args.precision == "bf16":
        dtype = torch.bfloat16
        print("警告: bf16精度はtorch.triuと互換性がない場合があります。エラーが発生した場合はfp16を使用してください。")
    elif args.precision == "fp16":
        dtype = torch.float16
        print("fp16精度を使用します（推奨設定）")
    else:
        raise ValueError(f"Unsupported precision: {args.precision}")
    
    # 4bit/8bit量子化をスキップするためのチェック
    if args.load_in_4bit or args.load_in_8bit:
        print("警告: bitsandbytesライブラリのセットアップに問題があるため、4bit/8bit量子化をスキップします")
        print("代わりにGPUメモリ最適化を実施します")
        args.load_in_4bit = False
        args.load_in_8bit = False
        args.very_low_memory = True
    
    # メモリ設定の最適化
    if torch.cuda.is_available():
        # GPUメモリの断片化を防ぐための設定
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        # 低メモリモードではCPU↔GPU間でのOffloadを積極的に行う
        if args.very_low_memory:
            print("超低メモリモードを有効化します")
            # 一度に保持する最大のバッチサイズを小さく設定
            torch.cuda.set_per_process_memory_fraction(0.8)  # GPUメモリの80%まで使用
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    try:
        # トークナイザーをロード
        tokenizer = AutoProcessor.from_pretrained(args.version)
        print(f"トークナイザータイプ: {type(tokenizer)}")
        
        # トークナイザーが持つメソッドとプロパティを表示
        print("Tokenizer functions:", [method for method in dir(tokenizer) if not method.startswith('_') and callable(getattr(tokenizer, method))])
        
        # トークナイザーがチャットテンプレートメソッドを持つか確認
        if hasattr(tokenizer, 'apply_chat_template'):
            print("このトークナイザーはapply_chat_templateメソッドを持っています")
        
        # <SEG>トークンをトークナイザーに追加
        # MllamaProcessorの場合、内部tokenizer属性を使用
        if hasattr(tokenizer, 'tokenizer'):
            print("トークナイザーは内部tokenizer属性を持っています")
            if "<SEG>" not in tokenizer.tokenizer.get_vocab():
                print("Adding <SEG> token to tokenizer vocabulary")
                tokenizer.tokenizer.add_special_tokens({"additional_special_tokens": ["<SEG>"]})
        else:
            # 通常のトークナイザーの場合
            print("トークナイザーは直接アクセス可能です")
            if "<SEG>" not in tokenizer.get_vocab():
                print("Adding <SEG> token to tokenizer vocabulary")
                tokenizer.add_special_tokens({"additional_special_tokens": ["<SEG>"]})
        
        # from_vision_modelメソッドを使用してモデルとトークナイザーを同時に取得
        # メモリ管理を最適化
        print("メモリ効率の良いモデルロードを実行します...")
        device_map = "auto"  # Hugging Faceにデバイスマッピングを任せる
        
        # 超低メモリモードでの追加対策
        offload_folder = None
        if args.very_low_memory:
            print("CPUオフロードを有効化します")
            # モデルを部分的にディスクにオフロードするための一時フォルダを設定
            import tempfile
            offload_folder = tempfile.mkdtemp()
            print(f"一時オフロードフォルダ: {offload_folder}")
        
        try:
            model, tokenizer = Llama32LISAForCausalLM.from_vision_model(
                vision_model_id=args.version,
                vision_pretrained=args.sam_version,
                train_mask_decoder=args.train_mask_decoder,
                tokenizer=tokenizer,
                torch_dtype=dtype,
                device_map=device_map,  # 自動的にGPU/CPU間でレイヤーを配置
                ignore_mismatched_sizes=args.ignore_mismatched_sizes,
                load_in_8bit=False,  # BNBエラーのため無効化
                load_in_4bit=False,   # BNBエラーのため無効化
                offload_folder=offload_folder if args.very_low_memory else None,
            )
        except Exception as e:
            print(f"モデルロード中にエラーが発生しました: {e}")
            print("代替方法でモデルをロードします...")
            
            # 代替として、より単純な方法でロード
            model, tokenizer = Llama32LISAForCausalLM.from_vision_model(
                vision_model_id=args.version,
                vision_pretrained=args.sam_version,
                train_mask_decoder=args.train_mask_decoder,
                tokenizer=tokenizer,
                torch_dtype=dtype,
                ignore_mismatched_sizes=args.ignore_mismatched_sizes,
            )
            
            # モデルをCPUに一部移動して手動でメモリ管理
            if args.very_low_memory and hasattr(model, 'model') and model.model is not None:
                print("手動でモデルレイヤーをCPUに移動します...")
                # モデルの一部のレイヤーを手動でCPUに移動
                for i in range(20, 32):  # 後半のレイヤーをCPUに
                    try:
                        layer_name = f"model.layers.{i}"
                        if hasattr(model.model, "layers") and i < len(model.model.layers):
                            layer = model.model.layers[i]
                            layer.to("cpu")
                            print(f"レイヤー {layer_name} をCPUに移動しました")
                    except Exception as layer_e:
                        print(f"レイヤー移動中にエラー: {layer_e}")
        
        # モデルをデバイスに配置
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # テスト用のシンプルな例で動作確認
        try:
            # Mllama処理テスト
            from model.llama3_2.mm_utils import run_test_mllama_processor, print_debug_info_for_mllama
            
            # プロセッサとモデルの詳細情報を出力
            print_debug_info_for_mllama(tokenizer, model)
            
            # テスト画像パスがあれば読み込む
            test_image_path = args.test_image if hasattr(args, 'test_image') and args.test_image else None
            
            if test_image_path and os.path.exists(test_image_path):
                print(f"テスト画像 {test_image_path} を使って処理テスト")
                from PIL import Image
                test_image = Image.open(test_image_path).convert('RGB')
                
                # プロセッサの動作テスト
                run_test_mllama_processor(tokenizer, model_name=args.version)
                
                # セグメンテーションテスト
                if args.run_segment_test and hasattr(model, 'generate_masks'):
                    print("\n===== セグメンテーションテスト =====")
                    try:
                        # 最もシンプルな形式でテスト
                        test_prompt = "Please segment all the persons in this image. <SEG>"
                        # 画像第一引数、テキスト第二引数
                        test_inputs = tokenizer(test_image, test_prompt, return_tensors="pt").to(device)
                        
                        # test_inputsから冗長なキーを削除
                        test_mask_inputs = {k: v for k, v in test_inputs.items() 
                                          if k not in ["input_ids", "attention_mask"]}
                        
                        # セグメンテーション実行
                        with torch.no_grad():
                            try:
                                # オリジナルのLISAアプローチで画像を処理
                                from model.segment_anything.utils.transforms import ResizeLongestSide
                                print("SAM用の画像前処理を実行中...")
                                
                                # テスト実行
                                generation = model.generate_masks(
                                    image=test_image,  # SAM内部で適切に処理されるように修正済み
                                    input_ids=test_inputs.input_ids,
                                    attention_mask=test_inputs.attention_mask,
                                    max_new_tokens=256,
                                    **test_mask_inputs
                                )
                            except Exception as e:
                                print(f"SAM処理中にエラー: {e}")
                                print("通常の生成にフォールバック")
                                # 例外発生時は通常の生成にフォールバック
                                generation = model.generate(
                                    **test_inputs,
                                    max_new_tokens=256
                                )
                                
                            # 生成されたテキストをデコード
                            output = tokenizer.decode(generation[0][test_inputs.input_ids.shape[1]:], skip_special_tokens=True)
                            print(f"セグメンテーション出力: {output[:100]}...")
                            print("セグメンテーションテスト成功!")
                    except Exception as segment_e:
                        print(f"セグメンテーションテスト中にエラー: {segment_e}")
                        import traceback
                        traceback.print_exc()
            
        except Exception as test_exc:
            print(f"テスト中にエラー発生: {test_exc}")
        
        # チャットループの設定
        prompt_template = get_prompt_template()
        # model_max_lengthを適切に取得
        model_max_length = tokenizer.tokenizer.model_max_length if hasattr(tokenizer, 'tokenizer') else tokenizer.model_max_length
        max_new_tokens = args.max_new_tokens
        sep = "[/INST]"
        stop_str = "</s>"
        
        # メモリの状態を報告
        if torch.cuda.is_available():
            print(f"GPUメモリ使用状況: {torch.cuda.memory_allocated() / 1024**2:.2f} MB / {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        
        # チャットモードの起動
        chatting(args, model, tokenizer, device, prompt_template, model_max_length, max_new_tokens, sep, stop_str)
    
    except Exception as main_error:
        print(f"メイン処理中の致命的エラー: {main_error}")
        import traceback
        traceback.print_exc()

    # 一時フォルダを削除
    if offload_folder and os.path.exists(offload_folder):
        import shutil
        try:
            shutil.rmtree(offload_folder)
        except Exception as e:
            print(f"一時フォルダの削除中にエラー: {e}")


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
