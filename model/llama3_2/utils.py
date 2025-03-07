"""
Llama3.2 Vision 用のユーティリティ関数
"""

import torch
import numpy as np
from PIL import Image

# Llama3.2 Vision の特殊トークン
BEGIN_OF_TEXT_TOKEN = "<|begin_of_text|>"
IMAGE_TOKEN = "<|image|>"
ASSISTANT_TOKEN = "<|start_header_id|>assistant<|end_header_id|>"
USER_TOKEN = "<|start_header_id|>user<|end_header_id|>"
EOT_TOKEN = "<|eot_id|>"

# セグメンテーション用の特殊トークン
SEG_TOKEN = "[SEG]"

def preprocess_image_for_llama32(image, processor):
    """
    画像をLlama3.2 Vision用に前処理
    
    Args:
        image: PIL.Image または numpy.ndarray
        processor: AutoProcessor
    
    Returns:
        processor の出力
    """
    # numpy配列の場合はPIL.Imageに変換
    if isinstance(image, np.ndarray):
        # BGRからRGBに変換
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = image[..., ::-1]
        image = Image.fromarray(image)
    
    # プロセッサで処理
    return processor(images=image, return_tensors="pt")

def preprocess_image_for_sam(image, transform, img_size=1024):
    """
    画像をSAM用に前処理
    
    Args:
        image: numpy.ndarray (BGR)
        transform: ResizeLongestSide
        img_size: ターゲットサイズ
    
    Returns:
        前処理された画像テンソル, リサイズサイズ, 元サイズ
    """
    # BGRからRGBに変換
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = image[..., ::-1]
    else:
        image_rgb = image
    
    # 元のサイズを記録
    original_size = image_rgb.shape[:2]
    
    # 変換
    resized_image = transform.apply_image(image_rgb)
    resize_size = resized_image.shape[:2]
    
    # 正規化とパディング
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    
    img_tensor = torch.from_numpy(resized_image).permute(2, 0, 1).contiguous()
    img_tensor = (img_tensor - pixel_mean) / pixel_std
    
    # パディング
    h, w = img_tensor.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    img_tensor = torch.nn.functional.pad(img_tensor, (0, padw, 0, padh))
    
    return img_tensor.unsqueeze(0), resize_size, original_size

def combine_text_and_image(text, processor):
    """
    テキストをLlama3.2 Visionの形式に変換
    
    Args:
        text: 入力テキスト
        processor: AutoProcessor
    
    Returns:
        プロセッサの出力
    """
    # Llama3.2 Visionフォーマットに変換
    prompt = f"{BEGIN_OF_TEXT_TOKEN}{IMAGE_TOKEN}{text}"
    return processor(text=prompt, return_tensors="pt")

def format_prompt(text, with_image=True):
    """
    プロンプトをLlama3.2 Vision用にフォーマット
    
    Args:
        text: テキストプロンプト
        with_image: 画像付きかどうか
    
    Returns:
        フォーマットされたプロンプト
    """
    if with_image:
        return f"{BEGIN_OF_TEXT_TOKEN}{IMAGE_TOKEN}{text}"
    else:
        return f"{BEGIN_OF_TEXT_TOKEN}{text}"

def process_seg_output(output_text):
    """
    出力テキストから[SEG]トークンの有無を確認
    
    Args:
        output_text: モデルの出力テキスト
    
    Returns:
        [SEG]を含むかどうか, [SEG]の位置
    """
    if SEG_TOKEN in output_text:
        seg_position = output_text.find(SEG_TOKEN)
        return True, seg_position
    return False, -1 