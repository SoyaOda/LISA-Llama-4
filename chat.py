import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor, AutoProcessor
from huggingface_hub import login

from model.LISA import LISAForCausalLM, Llama32LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.llama3_2 import conversation as llama3_2_conversation
from model.llama3_2.mm_utils import tokenizer_image_token as llama3_2_tokenizer_image_token
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
    parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v1")
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
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2", "llama3_2"],
    )
    parser.add_argument(
        "--sam_version",
        default=None,
        type=str,
        help="SAM model version, e.g., sam_vit_h"
    )
    parser.add_argument(
        "--model_type",
        default="lisa",
        type=str,
        choices=["lisa", "llama32_lisa"],
        help="Model type: original LISA or Llama3.2 Vision + SAM",
    )
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def main(args):
    # Hugging Face認証
    authenticate_huggingface()
    
    # 出力ディレクトリの作成
    if not os.path.exists(args.vis_save_path):
        os.makedirs(args.vis_save_path)

    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデル精度の設定
    if args.precision == "fp32":
        dtype = torch.float32
    elif args.precision == "bf16":
        dtype = torch.bfloat16
    elif args.precision == "fp16":
        dtype = torch.float16
    else:
        raise ValueError(f"Unsupported precision: {args.precision}")
    
    # 量子化の設定
    if args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
        )
    else:
        quantization_config = None

    # モデルタイプとバージョンに基づいてモデルを初期化
    if args.model_type == "lisa":
        # オリジナルLISAモデル
        print(f"Loading LISA model: {args.version}")
        tokenizer = AutoTokenizer.from_pretrained(args.version)
        model = LISAForCausalLM.from_pretrained(
            args.version,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None,
            quantization_config=quantization_config,
            torch_dtype=dtype,
            vision_tower=args.vision_tower,
            seg_token_idx=tokenizer.encode("<SEG>", add_special_tokens=False)[0],
        )
        image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", args.use_mm_start_end)
        
        # 会話タイプの設定
        if args.conv_type == "llava_v1":
            conv = conversation_lib.conv_templates["llava_v1"].copy()
        elif args.conv_type == "llava_llama_2":
            conv = conversation_lib.conv_templates["llava_llama_2"].copy()
        else:
            conv = conversation_lib.conv_templates["llava_v1"].copy()
    
    elif args.model_type == "llama32_lisa":
        # Llama3.2 Vision + SAMモデル
        print(f"Loading Llama3.2 Vision + SAM model: {args.version}")
        
        # Llama3.2 Visionモデル用のプロセッサ
        processor = AutoProcessor.from_pretrained(args.version)
        tokenizer = processor.tokenizer
        
        # <SEG>トークンがボキャブラリにない場合は追加
        if SEG_TOKEN not in tokenizer.get_vocab():
            print(f"Adding {SEG_TOKEN} token to tokenizer vocabulary")
            tokenizer.add_special_tokens({"additional_special_tokens": [SEG_TOKEN]})
        
        # SEGトークンのインデックスを取得
        seg_token_idx = tokenizer.convert_tokens_to_ids(SEG_TOKEN)
        
        # SAMのViT-Hチェックポイントパス
        sam_checkpoint = args.sam_version if args.sam_version else None
        
        # モデルをロード
        model = Llama32LISAForCausalLM.from_pretrained(
            args.version,
            vision_pretrained=sam_checkpoint,
            train_mask_decoder=False,   # 推論モード（SAMマスクデコーダを訓練しない）
            out_dim=256,                # マスク埋め込み次元（訓練設定と一致する必要あり）
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            quantization_config=quantization_config,
            seg_token_idx=seg_token_idx,
        )
        
        # トークナイザの語彙サイズをリサイズ（新しい特殊トークンを追加した場合）
        model.resize_token_embeddings(len(tokenizer))
        
        # 会話テンプレート
        if args.conv_type == "llama3_2":
            conv = llama3_2_conversation.conv_templates["llama3_2"].copy()
        else:
            # デフォルトはllama3_2を使用
            conv = llama3_2_conversation.conv_templates["llama3_2"].copy()
    
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    # モデルを評価モードに設定
    model.eval()
    
    # ここから対話ループ
    resize_transform = ResizeLongestSide(args.image_size)
    print("LISA chat system is ready. Type 'exit' to exit.")
    
    while True:
        try:
            # ユーザー入力を取得
            user_input = input("User: ")
            if user_input.lower() == "exit":
                break
                
            # 画像パスが含まれているか確認
            image_path = None
            for word in user_input.split():
                if os.path.exists(word) and (word.endswith(".jpg") or word.endswith(".png") or word.endswith(".jpeg")):
                    image_path = word
                    user_input = user_input.replace(image_path, "")
                    break
            
            # 画像が指定されている場合、処理を実行
            if image_path:
                # 画像を読み込み、前処理を行う
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                if args.model_type == "lisa":
                    # オリジナルLISA用の処理
                    # 画像のリサイズと前処理
                    image_resized = resize_transform.apply_image(image)
                    resize_ratio = image_resized.shape[0] / image.shape[0]
                    
                    # 画像の入力形式を準備
                    clip_image = image_processor.preprocess(image_resized, return_tensors="pt")["pixel_values"][0]
                    clip_image = clip_image.to(model.device, dtype=dtype)
                    
                    # SAM用の画像前処理
                    sam_image = preprocess(torch.from_numpy(image_resized).permute(2, 0, 1).contiguous())
                    sam_image = sam_image.to(model.device, dtype=dtype)
                    
                    # メッセージを会話に追加
                    if mm_use_im_start_end:
                        user_input = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + user_input
                    else:
                        user_input = DEFAULT_IMAGE_TOKEN + "\n" + user_input
                    
                    conv.append_message(conv.roles[0], user_input)
                    conv.append_message(conv.roles[1], None)
                    
                    # プロンプトの作成とトークン化
                    prompt = conv.get_prompt()
                    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX)
                    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
                    
                    # 推論の実行
                    with torch.no_grad():
                        output_ids, pred_masks = model.evaluate(
                            clip_image,
                            sam_image,
                            input_ids,
                            resize_list=[resize_ratio],
                            original_size_list=[image.shape[:2]],
                            max_new_tokens=512,
                            tokenizer=tokenizer,
                        )
                    
                else:  # Llama3.2 Vision + SAM
                    # Llama3.2 Vision + SAM用の処理
                    # 画像の前処理（1024x1024サイズにリサイズ）
                    image_resized = resize_transform.apply_image(image)
                    original_size = image.shape[:2]  # (height, width)
                    
                    # 画像をPIL形式に変換（プロセッサ用）
                    from PIL import Image as PILImage
                    pil_image = PILImage.fromarray(image_resized)
                    
                    # メッセージを会話に追加（<|image|>トークンを使用）
                    user_input = IMAGE_TOKEN + " " + user_input
                    conv.append_message(conv.roles[0], user_input)
                    conv.append_message(conv.roles[1], None)
                    
                    # プロンプトの作成
                    prompt = conv.get_prompt()
                    
                    # プロセッサで画像とテキストを前処理
                    inputs = processor(
                        images=pil_image,
                        text=prompt,
                        return_tensors="pt"
                    )
                    
                    # すべての入力をデバイスに移動
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    # 推論の実行
                    with torch.no_grad():
                        # 生成
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=512,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            output_hidden_states=True,
                            return_dict_in_generate=True,
                        )
                        
                        # 出力テキストをデコード
                        output_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
                        
                        # <SEG>トークンを含む場合、マスクを生成
                        if SEG_TOKEN in output_text:
                            # <SEG>トークンの位置を特定
                            seg_indices = []
                            for i, token_id in enumerate(outputs.sequences[0]):
                                if token_id == seg_token_idx:
                                    seg_indices.append(i)
                            
                            if seg_indices:
                                # SAM用の画像をテンソルに変換
                                sam_image = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).unsqueeze(0).float()
                                sam_image = sam_image.to(model.device, dtype=dtype)
                                
                                # マスクを生成
                                masks = model.generate_masks(
                                    images=sam_image,
                                    input_ids=outputs.sequences,
                                    seg_token_indices=seg_indices,
                                    original_sizes=[original_size],
                                )
                                
                                # マスクの可視化と保存
                                for i, mask in enumerate(masks):
                                    # マスクをCPUに移動してNumPy配列に変換
                                    mask_np = mask.cpu().numpy()
                                    # マスクを0-255の範囲にスケーリング
                                    mask_np = (mask_np * 255).astype(np.uint8)
                                    # マスクを元の画像サイズにリサイズ
                                    mask_np = cv2.resize(mask_np, (original_size[1], original_size[0]))
                                    
                                    # 画像とマスクのオーバーレイ
                                    overlay = image.copy()
                                    overlay[mask_np > 127] = [255, 0, 0]  # 赤色でマスク領域を表示
                                    overlay = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
                                    
                                    # 結果を保存
                                    save_path = os.path.join(args.vis_save_path, f"mask_{i}_{Path(image_path).stem}.png")
                                    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                                    print(f"マスク画像を保存しました: {save_path}")
                    
                # 会話の応答を更新
                if args.model_type == "lisa":
                    # オリジナルLISA用の応答処理
                    outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    conv.messages[-1][-1] = outputs
                    
                    # マスクが生成された場合、保存
                    if pred_masks:
                        for i, mask in enumerate(pred_masks):
                            # マスクをリサイズして画像と同じサイズに
                            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                            # マスクをオーバーレイとして適用
                            overlay = image.copy()
                            overlay[mask > 0.5] = [255, 0, 0]  # 赤色でマスク領域を表示
                            overlay = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
                            
                            # 結果を保存
                            save_path = os.path.join(args.vis_save_path, f"mask_{i}_{Path(image_path).stem}.png")
                            cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                            print(f"マスク画像を保存しました: {save_path}")
                            
                else:  # Llama3.2 Vision + SAM
                    # Llama3.2 Vision + SAM用の応答処理
                    # 応答テキストを整形（モデルの出力をユーザーフレンドリーにする）
                    response = output_text.split("[/INST]")[-1].strip()
                    response = response.replace(SEG_TOKEN, "*マスクを生成しました*")  # SEGトークンをユーザーフレンドリーなテキストに置換
                    
                    # 応答を会話に追加
                    conv.messages[-1][-1] = response
                
                # 応答を表示
                print(f"Assistant: {conv.messages[-1][-1]}")
                
            else:
                # 画像なしの通常の会話
                conv.append_message(conv.roles[0], user_input)
                conv.append_message(conv.roles[1], None)
                
                # プロンプトの作成とトークン化
                prompt = conv.get_prompt()
                
                if args.model_type == "lisa":
                    # オリジナルLISA用の処理
                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
                    
                    # 推論の実行
                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids=input_ids,
                            max_new_tokens=512,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                        )
                        
                    # 応答をデコード
                    outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                else:  # Llama3.2 Vision + SAM
                    # Llama3.2 Vision + SAM用の処理
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    
                    # 推論の実行
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=512,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                        )
                        
                    # 応答をデコード
                    outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # 応答テキストを整形
                    outputs = outputs.split("[/INST]")[-1].strip()
                
                # 応答を会話に追加
                conv.messages[-1][-1] = outputs
                
                # 応答を表示
                print(f"Assistant: {outputs}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
