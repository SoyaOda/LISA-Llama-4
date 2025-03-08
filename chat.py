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
# 会話ライブラリは必要に応じて動的にインポート
# from model.llava import conversation as conversation_lib
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
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
        help="Conversation template type",
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


def process_images(images, processor):
    """
    画像をモデル用に処理する関数
    """
    try:
        return processor(images, return_tensors="pt")
    except Exception as e:
        print(f"画像処理エラー: {e}")
        import traceback
        traceback.print_exc()
        # 代わりにnumpy処理を試す
        try:
            import numpy as np
            img_array = np.array(images[0])
            return {"pixel_values": torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0).float() / 255.0}
        except Exception as e2:
            print(f"バックアップ処理でもエラー: {e2}")
            return None

def create_segment_prompt(text):
    """
    セグメンテーション用のプロンプトを作成します
    Args:
        text: 入力テキスト
    Returns:
        SEGトークンを含むプロンプト
    """
    # SEGトークンが含まれているか確認し、なければ追加
    if SEG_TOKEN not in text:
        # 最後に追加
        if text.endswith('.') or text.endswith('?') or text.endswith('!'):
            text = f"{text} {SEG_TOKEN}"
        else:
            text = f"{text}. {SEG_TOKEN}"
        print(f"ユーザー入力にSEGトークンを追加しました")
    return text

def chatting(args, model, tokenizer, device, prompt_template, model_max_length, max_new_tokens, sep, stop_str):
    # チャットモードのパラメータ設定
    if hasattr(args, 'conv_type'):
        conv_mode = args.conv_type
    else:
        conv_mode = "llava_v1"  # デフォルト値を設定

    # Llama 3.2 Visionモデルの場合はLlama3.2会話形式を使用
    if args.version.startswith("meta-llama/Llama-3.2"):
        from model.llama3_2.conversation import conv_templates
        conv = conv_templates["llama3_2"].copy()
        print(f"Llama 3.2 Vision (Mllama)モデルを検出しました")
        print(f"画像トークン: {IMAGE_TOKEN}")
    else:
        # 通常のモデルではconversation_libを動的にインポート
        from model.llava import conversation as conversation_lib
        conv = conversation_lib.conv_templates[conv_mode].copy()
    
    # 会話履歴をリセット
    conv.messages = []
    
    # メモリ使用量に基づいた画像サイズの選択
    if args.low_memory:
        max_img_size = 256
        print(f"低メモリモード: 画像処理の最大サイズ: {max_img_size}x{max_img_size}")
    elif args.very_low_memory:
        max_img_size = 384
        print(f"超低メモリモード: 画像処理の最大サイズ: {max_img_size}x{max_img_size}")
    else:
        max_img_size = 768
        print(f"通常モード: 画像処理の最大サイズ: {max_img_size}x{max_img_size}")
    
    # メッセージ履歴の作成とチャットループ
    while True:
        try:
            # ユーザー入力の取得
            prompt = input("\nユーザー: ")
            if prompt.strip() == "exit":
                break
            
            # セグメンテーションのためのフラグ
            segment_mode = False
            
            # セグメンテーションが要求されているか確認
            if SEG_TOKEN in prompt or "segment" in prompt.lower() or "分割" in prompt or "抽出" in prompt:
                segment_mode = True
                # セグメンテーション用にプロンプトを準備
                prompt = create_segment_prompt(prompt)
                print(f"セグメンテーションモードで実行中...")
            
            # 画像の処理（画像パスが含まれている場合）
            if "http://" in prompt or "https://" in prompt or ".jpg" in prompt or ".png" in prompt or ".jpeg" in prompt:
                image_path = None
                
                if "http://" in prompt or "https://" in prompt:
                    image_regex = r'(https?://[^\s]+\.(?:jpg|jpeg|png|gif))'
                    image_paths = re.findall(image_regex, prompt)
                    
                    if image_paths:
                        image_path = image_paths[0]
                        # URLを画像トークンに置き換え
                        prompt = prompt.replace(image_path, IMAGE_TOKEN)
                    else:
                        print("有効な画像URLが見つかりませんでした")
                else:
                    # ローカルファイルパスを検索
                    image_regex = r'([^\s]+\.(?:jpg|jpeg|png|gif))'
                    image_paths = re.findall(image_regex, prompt)
                    
                    if image_paths:
                        image_path = image_paths[0]
                        # ファイルパスを画像トークンに置き換え
                        prompt = prompt.replace(image_path, IMAGE_TOKEN)
                    else:
                        print("有効な画像パスが見つかりませんでした")
            else:
                # 画像パスを明示的に尋ねる
                if IMAGE_TOKEN in prompt:
                    image_path = input("画像のパスを入力してください: ")
                else:
                    image_path = None
            
            # 画像の処理（画像パスがある場合）
            if image_path:
                try:
                    # 画像の読み込みと処理
                    image = Image.open(image_path)
                    
                    # 画像のリサイズ（メモリ使用量削減のため）
                    original_size = image.size
                    if original_size[0] > max_img_size or original_size[1] > max_img_size:
                        # アスペクト比を維持しながらリサイズ
                        image.thumbnail((max_img_size, max_img_size))
                        print(f"画像リサイズ: {original_size[0]}x{original_size[1]} -> {image.size[0]}x{image.size[1]}")
                    
                    # 会話にメッセージを追加
                    if args.version.startswith("meta-llama/Llama-3.2"):
                        # Llama 3.2 Visionモデルでは特別なフォーマットが必要
                        message = create_mllama_message(prompt, [image])
                        conv.append_message(conv.roles[0], message)
                    else:
                        # 通常のモデルではIMAGE_TOKENを置き換える
                        if IMAGE_TOKEN not in prompt:
                            prompt = IMAGE_TOKEN + "\n" + prompt
                            
                        if args.use_mm_start_end:
                            # 開始/終了トークンで画像トークンを囲む
                            replace_token = (
                                DEFAULT_IM_START_TOKEN + IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                            )
                            prompt = prompt.replace(IMAGE_TOKEN, replace_token)
                            
                        conv.append_message(conv.roles[0], prompt)
                    
                    # モデルの応答を追加するためのプレースホルダー
                    conv.append_message(conv.roles[1], "")
                    
                    # 最終的なプロンプトを生成
                    prompt = conv.get_prompt()
                    
                    # Llama 3.2 Visionモデルとその他のモデルで処理を分ける
                    if args.version.startswith("meta-llama/Llama-3.2"):
                        # MllamaProcessorのケース
                        # 画像処理をより詳細にデバッグ
                        print("Mllamaプロセッサを使用")
                        print(f"入力テキスト: {prompt[:50]}...")
                        try:
                            print("Mllamaプロセッサでtokenizer呼び出し前...")
                            print(f"画像タイプ: {type(image)}, テキスト: {prompt[:50]}...")
                            
                            # ProcessorでTokenizerと画像処理を同時に行う
                            inputs = tokenizer(
                                text=prompt,
                                images=[image],
                                return_tensors="pt"
                            )
                            
                            # デバッグ情報
                            for key, tensor in inputs.items():
                                print(f"inputs['{key}']の形状: {tensor.shape}, dtype: {tensor.dtype}")
                            
                            input_ids = inputs["input_ids"]
                            attention_mask = inputs["attention_mask"]
                            
                            # キューに追加
                            pixel_values = inputs["pixel_values"]
                            print(f"pixel_values形状: {pixel_values.shape}, dtype: {pixel_values.dtype}")
                            
                            # セグメンテーションモードの場合、追加のチェック
                            if segment_mode:
                                print(f"入力トークン数: {len(torch.where(input_ids[0] == tokenizer.tokenizer.convert_tokens_to_ids('<SEG>'))[0])}")
                                
                                # マスク生成に必要なパラメータを追加
                                mask_inputs = {}
                                for key, value in inputs.items():
                                    if key != "input_ids" and key != "attention_mask":
                                        mask_inputs[key] = value
                                
                                # デバイスに移動
                                input_ids = input_ids.to(device)
                                attention_mask = attention_mask.to(device)
                                
                                # 画像情報を表示
                                print(f"画像情報: {type(image)}, サイズ: {image.size}")
                                
                                # 画像ピクセル値の形状チェック
                                if 'pixel_values' in inputs and len(inputs['pixel_values'].shape) > 4:
                                    print("複雑な形状のpixel_valuesを検出しました。適切な形状に変換します...")
                                    pixel_values = inputs['pixel_values']
                                    
                                    try:
                                        # ユーティリティ関数を使用してSAM用に安全に変換
                                        from model.llama3_2.mm_utils import safe_mllama_to_sam
                                        pixel_values = safe_mllama_to_sam(pixel_values)
                                        print(f"SAM用に変換成功: 形状={pixel_values.shape}")
                                        
                                        # pixel_valuesキーを持つkwargsを作成
                                        # clean_mask_inputs = {k: v for k, v in mask_inputs.items() 
                                        #                   if k != 'pixel_values'}
                                        
                                        # トークナイザーを追加
                                        # clean_mask_inputs["tokenizer"] = tokenizer
                                        
                                        # generate_masksを実行
                                        result = model.generate_masks(
                                            image=None,  # PILを渡さない
                                            input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            pixel_values=pixel_values,  # 変換済みのテンソル
                                            max_new_tokens=max_new_tokens,
                                            do_sample=args.do_sample,
                                            temperature=args.temperature,
                                            top_p=args.top_p,
                                            tokenizer=tokenizer
                                        )
                                    except Exception as conversion_error:
                                        print(f"テンソル変換中にエラー: {conversion_error}")
                                        print("代わりにPIL画像を使用します...")
                                        
                                        # generate_masksを実行（PILイメージを使用）
                                        result = model.generate_masks(
                                            image=image,  # PIL Image
                                            input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            pixel_values=None,
                                            max_new_tokens=max_new_tokens,
                                            do_sample=args.do_sample,
                                            temperature=args.temperature,
                                            top_p=args.top_p,
                                            tokenizer=tokenizer
                                        )
                                else:
                                    # 通常の処理
                                    result = model.generate_masks(
                                        image=image,
                                        input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        max_new_tokens=max_new_tokens,
                                        do_sample=args.do_sample,
                                        temperature=args.temperature,
                                        top_p=args.top_p,
                                        tokenizer=tokenizer
                                    )
                                
                                # 結果の処理
                                if result is not None:
                                    # タプルの場合はトークンとマスクに分ける
                                    if isinstance(result, tuple) and len(result) == 2:
                                        generation = result[0]
                                        masks = result[1]
                                        print(f"生成されたマスク数: {len(masks) if masks else 0}")
                                        
                                        # マスクの保存
                                        if masks and len(masks) > 0:
                                            # マスクの保存場所
                                            os.makedirs(args.vis_save_path, exist_ok=True)
                                            
                                            for i, mask in enumerate(masks):
                                                try:
                                                    # マスクをNumPyに変換
                                                    mask_np = mask.detach().cpu().numpy()
                                                    
                                                    if mask_np.ndim > 2:
                                                        # 適切な次元を抽出
                                                        if mask_np.shape[0] == 1:
                                                            mask_np = mask_np[0]
                                                        
                                                        # 次元が3つ以上ある場合
                                                        if mask_np.ndim >= 3:
                                                            mask_np = mask_np[0]
                                                    
                                                    # バイナリマスクに変換
                                                    binary_mask = (mask_np > 0.5).astype(np.uint8) * 255
                                                    
                                                    # マスクとマスク適用画像の保存パス
                                                    mask_path = f"{args.vis_save_path}/{Path(image_path).stem}_mask_{i}.png"
                                                    masked_img_path = f"{args.vis_save_path}/{Path(image_path).stem}_masked_{i}.png"
                                                    
                                                    # マスクの保存
                                                    cv2.imwrite(mask_path, binary_mask)
                                                    print(f"マスクを保存しました: {mask_path}")
                                                    
                                                    # 元の画像にマスクを適用
                                                    img_np = np.array(Image.open(image_path))
                                                    
                                                    # マスクのリサイズ
                                                    if binary_mask.shape[:2] != img_np.shape[:2]:
                                                        binary_mask = cv2.resize(binary_mask, (img_np.shape[1], img_np.shape[0]))
                                                    
                                                    # マスクを適用した画像を生成
                                                    masked_img = img_np.copy()
                                                    masked_img[binary_mask > 0] = (
                                                        img_np[binary_mask > 0] * 0.5 + 
                                                        np.array([0, 0, 255]) * 0.5
                                                    ).astype(np.uint8)
                                                    
                                                    # BGR→RGB変換
                                                    if img_np.shape[-1] == 3:
                                                        masked_img = cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR)
                                                    
                                                    # マスク適用画像の保存
                                                    cv2.imwrite(masked_img_path, masked_img)
                                                    print(f"マスク適用画像を保存しました: {masked_img_path}")
                                                    
                                                except Exception as mask_save_error:
                                                    print(f"マスク保存中にエラー: {mask_save_error}")
                                                    import traceback
                                                    traceback.print_exc()
                                    else:
                                        # 通常の結果
                                        generation = result
                                        print("警告: マスクが生成されていません")
                                else:
                                    print("警告: 生成結果が見つかりません。エラーが発生した可能性があります。")
                                    generation = None
                            else:
                                # 通常のテキスト生成（セグメンテーションなし）
                                generation = model.generate(
                                    **inputs.to(device),
                                    max_new_tokens=max_new_tokens,
                                    do_sample=args.do_sample,
                                    temperature=args.temperature,
                                    top_p=args.top_p,
                                )
                        except Exception as process_error:
                            print(f"入力処理中にエラー: {process_error}")
                            import traceback
                            traceback.print_exc()
                            continue
                    else:
                        # 非MllamaProcessorのケース（他のモデル）
                        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").to(device)
                        
                        # 通常のテキスト生成
                        generation = model.generate(
                            input_ids=input_ids,
                            max_new_tokens=max_new_tokens,
                            do_sample=args.do_sample,
                            temperature=args.temperature,
                            top_p=args.top_p,
                        )
                    
                    # 出力テキストのデコード
                    try:
                        if generation is not None:
                            if args.version.startswith("meta-llama/Llama-3.2"):
                                # MllamaProcessor用のデコード処理
                                try:
                                    # input_idsの長さを考慮してスライスを作成
                                    input_length = input_ids.shape[1]
                                    
                                    # generationをデコード
                                    s = tokenizer.decode(generation[0][input_length:], skip_special_tokens=True)
                                    
                                    # 出力を整形
                                    s = s.strip()
                                except Exception as decode_error:
                                    print(f"Mllama生成結果のデコード中にエラー: {decode_error}")
                                    s = "デコード中にエラーが発生しました。もう一度お試しください。"
                            else:
                                # 通常のモデルのデコード処理
                                s = tokenizer.decode(generation[0])
                                
                                # 出力の整形
                                # 入力プロンプトの末尾を探して、それ以降のテキストを取得
                                if s.find(prompt) != -1:
                                    s = s[s.find(prompt) + len(prompt):]
                                # 特殊トークンを削除
                                s = s.replace("\n", "").replace("  ", " ")
                                
                            # 応答を会話に追加
                            conv.messages[-1][-1] = s
                            
                            # 出力テキストを整形して表示
                            print(f"\nAssistant: {s}")
                        else:
                            print("\nAssistant: エラーが発生しました。もう一度試してください。")
                    except Exception as decode_error:
                        print(f"出力デコード中にエラー: {decode_error}")
                        import traceback
                        traceback.print_exc()
                        print("\nAssistant: 結果のデコード中にエラーが発生しました。もう一度試してください。")
                except Exception as img_error:
                    print(f"画像処理中にエラー: {img_error}")
                    import traceback
                    traceback.print_exc()
            else:
                # 画像なしの通常のテキスト処理
                conv.append_message(conv.roles[0], prompt)
                conv.append_message(conv.roles[1], "")
                
                # 最終的なプロンプトを生成
                prompt = conv.get_prompt()
                
                # 入力IDの取得
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                # テキスト生成
                try:
                    generation = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p
                    )
                    
                    # 出力のデコード
                    try:
                        if args.version.startswith("meta-llama/Llama-3.2"):
                            # MllamaProcessor用のデコード処理
                            input_length = inputs.input_ids.shape[1]
                            s = tokenizer.decode(generation[0][input_length:], skip_special_tokens=True)
                        else:
                            # 通常のデコード処理
                            s = tokenizer.decode(generation[0])
                            if s.find(prompt) != -1:
                                s = s[s.find(prompt) + len(prompt):]
                        
                        # 応答を会話に追加して表示
                        conv.messages[-1][-1] = s
                        print(f"\nAssistant: {s}")
                    except Exception as decode_error:
                        print(f"出力デコード中にエラー: {decode_error}")
                        print("\nAssistant: 結果のデコード中にエラーが発生しました。もう一度試してください。")
                except Exception as gen_error:
                    print(f"テキスト生成中にエラー: {gen_error}")
                    print("\nAssistant: エラーが発生しました。もう一度試してください。")
        except Exception as e:
            print(f"チャット処理中に予期せぬエラーが発生: {e}")
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
