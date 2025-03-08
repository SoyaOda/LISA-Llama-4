import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoProcessor

from model.llama3_2 import Llama32SAMForCausalLM, Llama32SAMConfig
from model.llama3_2.hf_auth import login_huggingface
from model.llama3_2.processor_utils import prepare_llama32_prompt, postprocess_llama32_output
from model.segment_anything.utils.transforms import ResizeLongestSide
from model.llama3_2.utils import (
    BEGIN_OF_TEXT_TOKEN, 
    IMAGE_TOKEN, 
    preprocess_image_for_sam
)


def parse_args(args):
    parser = argparse.ArgumentParser(description="Llama3.2 Vision + SAM chat")
    parser.add_argument("--version", default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    parser.add_argument("--model_type", choices=["llama3_2"], default="llama3_2", 
                        help="Model type to use: llama3_2 (Llama3.2 Vision + SAM)")
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
    parser.add_argument("--vision_pretrained", default="./checkpoints/sam_vit_h_4b8939.pth", type=str,
                        help="Path to the SAM vision encoder checkpoint")
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--hf_token", default=None, type=str, 
                        help="Hugging Face token for downloading models")
    parser.add_argument("--prompt", type=str, help="Prompt text to use")
    parser.add_argument("--image_path", type=str, help="Path to the image")
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
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Hugging Faceにログイン
    if args.hf_token:
        login_huggingface(args.hf_token)

    # トークナイザーの設定
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    
    # [SEG]トークンをトークナイザーに追加
    if "[SEG]" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["[SEG]"])
    
    # [SEG]トークンのIDを取得
    args.seg_token_idx = tokenizer.convert_tokens_to_ids("[SEG]")

    # 精度の設定
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    # Llama3.2 + SAMモデルの初期化
    print(f"Loading Llama3.2 Vision + SAM model from {args.version}")
    
    # カスタム設定クラスの作成
    config = Llama32SAMConfig(
        model_id=args.version,
        precision=args.precision,
        train_mask_decoder=False,
        out_dim=256,  # SAMのプロンプトエンコーダの次元数
        hidden_size=4096,  # Llama3.2-11Bの隠れ次元
        vision_tower=args.vision_tower,
        mm_vision_tower=args.vision_tower,
        mm_use_im_start_end=args.use_mm_start_end,
        seg_token_idx=args.seg_token_idx,
        vision_pretrained=args.vision_pretrained
    )
    
    # モデルの初期化
    model = Llama32SAMForCausalLM(
        config, 
        seg_token_idx=args.seg_token_idx,
        vision_pretrained=args.vision_pretrained,
        **kwargs
    )

    # モデルの精度を設定
    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit):
        model = model.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    # 画像変換用のtransform
    transform = ResizeLongestSide(args.image_size)

    model.eval()
    
    print(f"Model loaded: {args.model_type} - {args.version}")
    
    # コマンドライン引数からプロンプトと画像パスが指定されている場合
    if args.prompt and args.image_path:
        prompt = args.prompt
        image_path = args.image_path
        
        if not os.path.exists(image_path):
            print(f"画像ファイルが見つかりません: {image_path}")
            return
            
        process_image_and_prompt(model, tokenizer, prompt, image_path, args, transform)
    else:
        # インタラクティブモード
        print("Ready for conversation. Type 'exit' to quit.")
        while True:
            prompt = input("Please input your prompt: ")
            if prompt.lower() == "exit":
                break

            image_path = input("Please input the image path: ")
            if not os.path.exists(image_path):
                print("File not found in {}".format(image_path))
                continue
                
            process_image_and_prompt(model, tokenizer, prompt, image_path, args, transform)

# 画像処理とプロンプト処理を行う関数を追加
def process_image_and_prompt(model, tokenizer, prompt, image_path, args, transform):
    # 画像を読み込み
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]

    # SAM用の画像前処理
    sam_tensor, resize_size, original_size = preprocess_image_for_sam(image_np, transform)
    sam_tensor = sam_tensor.cuda()
    
    # 精度に合わせて変換
    if args.precision == "bf16":
        sam_tensor = sam_tensor.bfloat16()
    elif args.precision == "fp16":
        sam_tensor = sam_tensor.half()
    else:
        sam_tensor = sam_tensor.float()

    # Llama3.2 Vision用の処理
    # 画像をプロセッサで処理
    image_inputs = model.processor(
        images=image_np, 
        return_tensors="pt"
    ).to("cuda")
    
    # 精度変換
    if args.precision == "bf16":
        image_inputs = {k: v.bfloat16() if isinstance(v, torch.Tensor) else v for k, v in image_inputs.items()}
    elif args.precision == "fp16":
        image_inputs = {k: v.half() if isinstance(v, torch.Tensor) else v for k, v in image_inputs.items()}

    # プロンプトをトークン化
    input_text = f"{BEGIN_OF_TEXT_TOKEN}{IMAGE_TOKEN}{prompt}"
    text_inputs = model.processor(
        text=input_text,
        return_tensors="pt"
    ).to("cuda")
    
    # 入力を結合
    combined_inputs = {
        **image_inputs,
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"] if "attention_mask" in text_inputs else None
    }
    
    # 推論
    print("Generating response...")
    output_ids, pred_masks = model.evaluate(
        image_inputs,
        sam_tensor,
        combined_inputs["input_ids"],
        [resize_size],
        original_size_list,
        max_new_tokens=512,
        tokenizer=tokenizer,
    )

    # テキスト出力の処理
    if hasattr(output_ids, 'sequences'):
        output_ids = output_ids.sequences[0]
    
    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    print("text_output: ", text_output)

    # マスク出力の処理
    for i, pred_mask in enumerate(pred_masks):
        if pred_mask.shape[0] == 0:
            continue

        pred_mask = pred_mask.detach().cpu().numpy()[0]
        pred_mask = pred_mask > 0

        save_path = "{}/{}_mask_{}.jpg".format(
            args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
        )
        cv2.imwrite(save_path, pred_mask * 100)
        print("{} has been saved.".format(save_path))

        save_path = "{}/{}_masked_img_{}.jpg".format(
            args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
        )
        save_img = image_np.copy()
        save_img[pred_mask] = (
            image_np * 0.5
            + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
        )[pred_mask]
        save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, save_img)
        print("{} has been saved.".format(save_path))


if __name__ == "__main__":
    main(sys.argv[1:])
