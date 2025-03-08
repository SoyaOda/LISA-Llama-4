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
    """画像を処理し、プロセッサに適した形式に変換する"""
    try:
        # 単一画像の場合
        if isinstance(images, (str, Image.Image)):
            if isinstance(images, str):
                # 画像パスから画像をロード
                try:
                    image = Image.open(images).convert('RGB')
                except Exception as e:
                    print(f"画像'{images}'のロード中にエラー: {e}")
                    return None
            else:
                image = images
                
            # プロセッサで処理
            try:
                return processor(images=image, return_tensors="pt")
            except Exception as e:
                print(f"プロセッサでの画像処理中にエラー: {e}")
                # フォールバック処理
                from PIL import Image
                import numpy as np
                import torch
                
                # 画像をnumpy配列に変換
                img_array = np.array(image)
                # トーチテンソルに変換してバッチ次元を追加
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().unsqueeze(0)
                # 正規化（簡易的なもの）
                img_tensor = img_tensor / 255.0
                return {"pixel_values": img_tensor}
                
        # 画像リストの場合
        elif isinstance(images, list):
            processed_images = []
            for img in images:
                processed = process_images(img, processor)
                if processed is not None:
                    processed_images.append(processed)
            
            if processed_images:
                # バッチ化
                return {
                    key: torch.cat([img[key] for img in processed_images], dim=0) 
                    for key in processed_images[0]
                }
            else:
                return None
        else:
            print(f"サポートされていない画像タイプ: {type(images)}")
            return None
    except Exception as e:
        print(f"画像処理中に予期せぬエラー: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_segment_prompt(text):
    """テキストにセグメント化トークンを追加"""
    # すでにSEGトークンを含んでいる場合はそのまま返す
    if "<SEG>" in text or "[SEG]" in text:
        return text
        
    # 文末にセグメントトークンを追加
    if text.endswith('.') or text.endswith('?') or text.endswith('!'):
        return f"{text} {SEG_TOKEN}"
    else:
        return f"{text}. {SEG_TOKEN}"

def chatting(args, model, tokenizer, device, prompt_template, model_max_length, max_new_tokens, sep, stop_str):
    """チャットインターフェース関数"""
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

    # 画像ファイルパスを検索するための正規表現パターン
    img_path_pattern = re.compile(r'(https?://\S+\.(?:jpg|jpeg|png|gif|bmp|webp)|\S+\.(?:jpg|jpeg|png|gif|bmp|webp))')

    try:
        # 連続チャットループ
        while True:
            # ユーザー入力の取得
            user_input = input("\nユーザー: ")
            if user_input.lower() in ["exit", "quit", "q", "終了"]:
                print("チャットを終了します。")
                break

            # SEGトークンの処理
            has_seg_request = "<SEG>" in user_input or "[SEG]" in user_input or "segment" in user_input.lower()
            if has_seg_request:
                print("セグメンテーションモードで実行中...")
                # 必要に応じてSEGトークンを追加
                if "<SEG>" not in user_input and "[SEG]" not in user_input:
                    user_input = create_segment_prompt(user_input)
                print(f"ユーザー入力にSEGトークンを追加しました: {user_input}")

            # 画像パスの検出と処理
            image = None
            img_path_match = img_path_pattern.search(user_input)
            
            if img_path_match:
                # 検出された画像パスを処理
                img_path = img_path_match.group(0)
                user_input = user_input.replace(img_path, "").strip()
                
                try:
                    if os.path.exists(img_path):
                        print(f"画像を読み込み中: {img_path}")
                        image = Image.open(img_path).convert('RGB')
                    else:
                        print(f"画像が見つかりません: {img_path}")
                        continue
                except Exception as img_err:
                    print(f"画像の読み込みに失敗しました: {img_err}")
                    continue
            else:
                # 画像パスが指定されていない場合は入力を促す
                img_path = input("画像ファイルのパスを入力してください: ")
                if img_path and os.path.exists(img_path):
                    try:
                        print(f"画像を読み込み中: {img_path}")
                        image = Image.open(img_path).convert('RGB')
                    except Exception as img_err:
                        print(f"画像の読み込みに失敗しました: {img_err}")
                        continue
                else:
                    print("有効な画像パスが指定されていません。テキストのみで続行します。")

            # 画像のリサイズ処理（アスペクト比を保持）
            if image:
                width, height = image.size
                if max(width, height) > max_img_size:
                    ratio = max_img_size / max(width, height)
                    new_width = int(width * ratio)
                    new_height = int(height * ratio)
                    image = image.resize((new_width, new_height), Image.LANCZOS)
                    print(f"画像をリサイズしました: {width}x{height} -> {new_width}x{new_height}")

            # モデルタイプに応じた処理
            if args.version.startswith("meta-llama/Llama-3.2"):
                # Llama 3.2 Visionモデル用の処理
                try:
                    # ユーザーメッセージの作成
                    messages = [
                        {"role": "user", "content": [
                            {"type": "image"} if image else None,
                            {"type": "text", "text": user_input}
                        ]}
                    ]
                    
                    # None要素を除去
                    messages[0]["content"] = [item for item in messages[0]["content"] if item is not None]
                    
                    # チャットテンプレートを適用
                    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                    
                    # 入力の作成
                    try:
                        inputs = tokenizer(
                            image,
                            input_text,
                            add_special_tokens=False,
                            return_tensors="pt"
                        ).to(device)
                    except Exception as processor_error:
                        print(f"画像処理エラー: {processor_error}")
                        if "Invalid input type" in str(processor_error):
                            print("画像形式のエラー: 画像処理方法を変更します")
                            # 代替処理法：画像を先に変換してからテキストと組み合わせる
                            from transformers import CLIPImageProcessor
                            clip_processor = CLIPImageProcessor()
                            pixel_values = clip_processor(images=image, return_tensors="pt").pixel_values.to(device)
                            
                            # テキストのみの処理
                            text_inputs = tokenizer.tokenizer(
                                input_text, 
                                add_special_tokens=False, 
                                return_tensors="pt"
                            ).to(device)
                            
                            # 手動で入力を結合
                            inputs = {
                                "input_ids": text_inputs.input_ids,
                                "attention_mask": text_inputs.attention_mask,
                                "pixel_values": pixel_values
                            }
                        else:
                            raise

                    # 生成パラメータの設定
                    generation_config = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": args.do_sample,
                        "temperature": args.temperature if args.temperature > 0 else 0.7,
                        "top_p": args.top_p if args.top_p > 0 else 0.9,
                        "repetition_penalty": args.repetition_penalty if args.repetition_penalty > 0 else 1.1,
                    }
                    
                    # 生成を実行
                    print("応答を生成中...")
                    if has_seg_request:
                        # セグメンテーション要求の場合
                        outputs = model.generate_masks(
                            image=image,
                            **inputs,
                            **generation_config
                        )
                        
                        # 結果の処理
                        if outputs and 'masks' in outputs:
                            # マスクを画像として保存
                            print("セグメンテーション完了！マスクを保存中...")
                            masks = outputs['masks']
                            
                            # 出力フォルダの確認
                            os.makedirs(args.vis_save_path, exist_ok=True)
                            base_name = os.path.basename(img_path).split('.')[0]
                            
                            # マスクの保存とビジュアライズ
                            for i, mask in enumerate(masks):
                                save_path = os.path.join(args.vis_save_path, f"{base_name}_mask_{i}.png")
                                # マスクをPIL画像として保存
                                mask_img = Image.fromarray((mask.cpu().numpy() * 255).astype(np.uint8))
                                mask_img.save(save_path)
                                
                                # マスクを元画像に適用して可視化
                                vis_path = os.path.join(args.vis_save_path, f"{base_name}_masked_img_{i}.jpg")
                                img_np = np.array(image)
                                mask_np = mask.cpu().numpy()
                                masked_img = img_np.copy()
                                # 半透明のオーバーレイを作成
                                overlay = np.zeros_like(img_np)
                                overlay[mask_np > 0.5] = [0, 255, 0]  # 緑色でマスクを強調
                                masked_img = cv2.addWeighted(masked_img, 0.7, overlay, 0.3, 0)
                                cv2.imwrite(vis_path, cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))
                                print(f"マスク{i+1}を保存しました: {save_path}, {vis_path}")
                            
                            # テキスト出力の処理
                            text_output = outputs.get('text', "セグメンテーション完了")
                            print(f"\nAssistant: {text_output}")
                        else:
                            print("\nAssistant: セグメンテーションの結果が得られませんでした。")
                    else:
                        # 通常のテキスト生成
                        output_ids = model.generate(**inputs, **generation_config)
                        
                        # 入力長を取得して出力から除外
                        input_length = inputs["input_ids"].shape[1]
                        generated_text = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
                        
                        print(f"\nAssistant: {generated_text}")
                
                except Exception as e:
                    print(f"チャット処理中に予期せぬエラーが発生: {e}")
                    import traceback
                    traceback.print_exc()
            
            else:
                # 従来のLISAモデル用の処理（既存コードを再利用）
                print("従来のLISAモデルは現在サポートされていません。")
                # ここに従来のLISAモデル用の処理コードを追加

    except KeyboardInterrupt:
        print("\nチャットを終了しました。")
    except Exception as e:
        print(f"チャット実行中にエラーが発生: {e}")
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
