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
    """
    チャットループを実行する関数
    """
    # 会話履歴の初期化
    if args.conv_type == "llama3_2":
        from model.llama3_2.conversation import get_default_conv_template
        conv = get_default_conv_template()
    else:
        from model.llava.conversation import conv_templates
        conv = conv_templates["llava_v1"].copy()
    
    # メモリ使用量を表示
    if torch.cuda.is_available():
        print(f"チャット開始時のGPUメモリ使用量: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        torch.cuda.empty_cache()
    
    # 画像サイズの設定
    # メモリ使用量を削減するために小さいサイズを使用
    if args.very_low_memory:
        max_image_size = 256  # 超低メモリモードでは256x256に制限
    elif args.low_memory:
        max_image_size = 384  # 低メモリモードでは384x384に制限
    else:
        max_image_size = 512  # 通常モードでは512x512に制限
    
    print(f"画像サイズを{max_image_size}x{max_image_size}に制限します")
    
    # チャットループ
    while True:
        try:
            # ユーザー入力の取得
            prompt = input("\n- Please input your prompt: ")
            if prompt == "exit":
                break
            
            # SEGトークンの処理
            prompt = create_segment_prompt(prompt)
            
            # 画像パスの取得
            image_path = input("- Please input the image path: ")
            if not os.path.exists(image_path):
                print(f"画像が見つかりません: {image_path}")
                continue
            
            # 画像の読み込みと前処理
            try:
                # PILで画像を読み込む
                image = Image.open(image_path).convert('RGB')
                
                # メモリ使用量削減のために画像をリサイズ
                # アスペクト比を維持しながら最大サイズを制限
                width, height = image.size
                if max(width, height) > max_image_size:
                    # アスペクト比を維持しながらリサイズ
                    if width > height:
                        new_width = max_image_size
                        new_height = int(height * (max_image_size / width))
                    else:
                        new_height = max_image_size
                        new_width = int(width * (max_image_size / height))
                    
                    print(f"画像をリサイズします: {width}x{height} -> {new_width}x{new_height}")
                    image = image.resize((new_width, new_height), Image.LANCZOS)
                
                # メモリ使用量を表示
                if torch.cuda.is_available():
                    print(f"画像読み込み後のGPUメモリ使用量: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
                
                # 画像処理
                try:
                    # MllamaProcessorを使用して画像を処理
                    processed_images = process_images(image, tokenizer)
                    
                    # メッセージの作成
                    messages = [
                        {"role": "user", "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt}
                        ]}
                    ]
                    
                    # チャットテンプレートの適用
                    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                    
                    # 入力の処理
                    inputs = tokenizer(
                        text=input_text,
                        images=processed_images,
                        return_tensors="pt",
                        padding=True
                    )
                    
                    # デバイスに移動
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # メモリ使用量を表示
                    if torch.cuda.is_available():
                        print(f"入力処理後のGPUメモリ使用量: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
                        torch.cuda.empty_cache()
                    
                    # マスク生成のための入力を準備
                    mask_inputs = {
                        "tokenizer": tokenizer,
                        "image": image,  # オリジナルサイズの画像
                    }
                    
                    # 生成パラメータの設定
                    gen_kwargs = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": args.do_sample,
                        "temperature": args.temperature if args.temperature > 0 else 0.7,
                        "top_p": args.top_p if args.top_p > 0 else 0.9,
                    }
                    
                    # マスク生成の実行
                    print("マスク生成を開始します...")
                    with torch.no_grad():  # メモリ使用量削減のためgrad不要モードで実行
                        outputs = model.generate_masks(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            pixel_values=inputs.get("pixel_values"),
                            **mask_inputs,
                            **gen_kwargs
                        )
                    
                    # メモリ使用量を表示
                    if torch.cuda.is_available():
                        print(f"生成後のGPUメモリ使用量: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
                        torch.cuda.empty_cache()
                    
                    # 出力の処理
                    if "error" in outputs:
                        print(f"エラーが発生しました: {outputs['error']}")
                        continue
                    
                    # 生成されたテキストの表示
                    generated_text = outputs.get("generated_text", "")
                    if generated_text:
                        print("\n" + "="*50)
                        print("生成されたテキスト:")
                        print(generated_text)
                        print("="*50)
                    
                    # マスクの処理と保存
                    masks = outputs.get("masks", [])
                    if masks:
                        print(f"\n{len(masks)}個のマスクが生成されました")
                        
                        # マスクの可視化と保存
                        for i, mask in enumerate(masks):
                            try:
                                # マスクをCPUに移動
                                mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
                                
                                # 画像にマスクを適用
                                image_np = np.array(image)
                                masked_img = image_np.copy()
                                
                                # マスクのサイズが画像と異なる場合はリサイズ
                                if mask_np.shape[:2] != image_np.shape[:2]:
                                    from skimage.transform import resize
                                    mask_np = resize(mask_np, image_np.shape[:2], order=0, preserve_range=True)
                                
                                # マスクを適用（半透明の色付きオーバーレイ）
                                mask_color = np.array([30, 144, 255])  # 青色
                                masked_img[mask_np > 0.5] = masked_img[mask_np > 0.5] * 0.7 + mask_color * 0.3
                                
                                # 保存
                                output_path = os.path.join(
                                    args.vis_save_path, 
                                    f"{os.path.basename(image_path).split('.')[0]}_masked_img_{i}.jpg"
                                )
                                Image.fromarray(masked_img.astype(np.uint8)).save(output_path)
                                print(f"マスク付き画像を保存しました: {output_path}")
                            except Exception as mask_error:
                                print(f"マスク処理中にエラー: {mask_error}")
                    else:
                        print("マスクは生成されませんでした")
                    
                except Exception as proc_error:
                    print(f"画像処理中にエラー: {proc_error}")
                    import traceback
                    traceback.print_exc()
            
            except Exception as img_error:
                print(f"画像読み込み中にエラー: {img_error}")
        
        except KeyboardInterrupt:
            print("\nチャットを終了します")
            break
        
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
    
    # メモリ使用量を表示
    if torch.cuda.is_available():
        print(f"チャット終了時のGPUメモリ使用量: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        torch.cuda.empty_cache()


def main(args):
    # Hugging Face認証
    if args.authenticate:
        authenticate_huggingface()
    
    # 一時フォルダの設定
    offload_folder = None
    if args.offload_folder:
        offload_folder = args.offload_folder
        os.makedirs(offload_folder, exist_ok=True)
        print(f"オフロードフォルダを作成しました: {offload_folder}")
    
    # メモリ使用量を表示
    if torch.cuda.is_available():
        print(f"初期GPUメモリ使用量: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"利用可能なGPUメモリ: {torch.cuda.get_device_properties(0).total_memory/1024**2:.2f} MB")
    
    # メモリ使用量を削減するための設定
    # very_low_memoryモードを強制的に有効化
    args.very_low_memory = True
    print("メモリ使用量削減のためにvery_low_memoryモードを有効化しました")
    
    # 最大画像サイズをさらに小さく設定
    max_image_size = 256  # very_low_memoryモードでの最大サイズ
    print(f"画像サイズを{max_image_size}x{max_image_size}に制限します")
    
    try:
        # モデル設定
        print(f"モデルをロード中: {args.version}")
        
        # 量子化設定
        quantization_config = None
        if args.load_in_8bit:
            print("8ビット量子化を使用します")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        elif args.load_in_4bit:
            print("4ビット量子化を使用します")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # デバイス設定
        device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
        print(f"使用デバイス: {device}")
        
        # データ型設定
        torch_dtype = torch.float32
        if args.precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
            print("BFloat16精度を使用します")
        elif args.precision == "fp16":
            torch_dtype = torch.float16
            print("Float16精度を使用します")
        else:
            print("Float32精度を使用します")
        
        # トークナイザーのロード
        tokenizer = AutoTokenizer.from_pretrained(
            args.version,
            trust_remote_code=args.trust_remote_code
        )
        
        # SAMチェックポイントの設定
        sam_checkpoint = args.sam_checkpoint
        if not sam_checkpoint or not os.path.exists(sam_checkpoint):
            print(f"警告: SAMチェックポイントが見つかりません: {sam_checkpoint}")
            print("デフォルトのSAMモデルを使用します")
            sam_checkpoint = None
        
        # モデルのロード
        print("モデルをロード中...")
        
        # メモリ使用量を監視
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"モデルロード前のGPUメモリ使用量: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        
        # モデルをロード
        model = Llama32LISAForCausalLM.from_vision_model(
            args.version,
            vision_pretrained=sam_checkpoint,
            train_mask_decoder=args.train_mask_decoder,
            tokenizer=tokenizer,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
            quantization_config=quantization_config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            offload_folder=offload_folder
        )
        
        # メモリ使用量を表示
        if torch.cuda.is_available():
            print(f"モデルロード後のGPUメモリ使用量: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        
        # モデルを評価モードに設定
        model.eval()
        
        # 会話設定
        if args.conv_type == "llava_v1":
            from model.llava.conversation import conv_templates
            conv_mode = "llava_v1"
        elif args.conv_type == "llama3_2":
            from model.llama3_2.conversation import get_default_conv_template
            conv_mode = "llama3_2"
        else:
            raise ValueError(f"未知の会話タイプ: {args.conv_type}")
        
        # プロンプトテンプレートの取得
        prompt_template = get_prompt_template()
        
        # モデルの最大長とトークン設定
        model_max_length = None
        if hasattr(model, 'config') and hasattr(model.config, 'max_position_embeddings'):
            model_max_length = model.config.max_position_embeddings
        elif hasattr(tokenizer, 'model_max_length'):
            model_max_length = tokenizer.model_max_length
        
        # 最大長が設定されていない場合はデフォルト値を使用
        if not model_max_length or model_max_length > 8192:
            model_max_length = 4096
            print(f"モデルの最大長をデフォルト値に設定: {model_max_length}")
        
        # 最大新トークン数の設定
        max_new_tokens = min(args.max_new_tokens, 1024)  # メモリ使用量削減のため制限
        print(f"最大新トークン数: {max_new_tokens}")
        
        # 区切り文字と停止文字列の設定
        sep = "\n"
        stop_str = None
        
        # チャット開始
        print("\n" + "="*50)
        print(f"チャットを開始します。終了するには 'exit' と入力してください。")
        print("="*50 + "\n")
        
        # チャット関数を呼び出し
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
