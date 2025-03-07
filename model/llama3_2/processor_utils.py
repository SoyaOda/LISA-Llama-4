import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor

from .utils import (
    BEGIN_OF_TEXT_TOKEN, 
    IMAGE_TOKEN, 
    SEG_TOKEN, 
    preprocess_image_for_llama32,
    format_prompt
)

def prepare_images_for_llama32(images, model_id="meta-llama/Llama-3.2-11B-Vision-Instruct"):
    """
    Llama3.2 Visionモデル用に画像を前処理
    Args:
        images: 画像（PILのImageオブジェクト、OpenCVのndarray、またはリスト）
        model_id: モデルID
    Returns:
        プロセッサの出力（pixel_values等）
    """
    processor = AutoProcessor.from_pretrained(model_id)
    
    # 単一画像の場合はリストに変換
    if not isinstance(images, list):
        images = [images]
    
    # OpenCV形式(BGR)からPIL形式(RGB)に変換
    processed_images = []
    for img in images:
        if isinstance(img, np.ndarray):
            if img.shape[2] == 3:  # Check if it's a color image
                img = img[..., ::-1]  # BGR to RGB
            img = Image.fromarray(img)
        processed_images.append(img)
    
    # プロセッサで処理
    return processor(images=processed_images, return_tensors="pt")

def prepare_llama32_prompt(text, image=None, model_id="meta-llama/Llama-3.2-11B-Vision-Instruct"):
    """
    Llama3.2 Vision モデル用にプロンプトを準備
    Args:
        text: 入力テキスト
        image: オプションの画像
        model_id: モデルID
    Returns:
        プロセッサの出力
    """
    processor = AutoProcessor.from_pretrained(model_id)
    
    if image is not None:
        # 画像付きの場合
        return processor(images=image, text=format_prompt(text, with_image=True), return_tensors="pt")
    else:
        # テキストのみの場合
        return processor(text=format_prompt(text, with_image=False), return_tensors="pt")

def convert_llava_to_llama32_format(llava_text, tokenizer=None):
    """
    LLaVAフォーマットのテキストをLlama3.2フォーマットに変換
    Args:
        llava_text: LLaVAフォーマットのテキスト
        tokenizer: Llama3.2のトークナイザ
    Returns:
        Llama3.2フォーマットのテキスト
    """
    # LLaVAの特殊トークンからLlama3.2の特殊トークンへの変換
    return format_prompt(llava_text)

def postprocess_llama32_output(output, processor):
    """
    Llama3.2の生成結果を後処理
    Args:
        output: モデルの出力
        processor: Llama3.2のプロセッサ
    Returns:
        処理済みテキスト
    """
    if hasattr(output, "sequences"):
        decoded = processor.decode(output.sequences[0], skip_special_tokens=True)
    else:
        decoded = processor.decode(output[0], skip_special_tokens=True)
    
    # <|eot_id|>などの特殊トークンを除去
    cleaned_text = decoded.replace("<|eot_id|>", "").strip()
    return cleaned_text 