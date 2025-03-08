import base64
from io import BytesIO
from typing import List, Optional, Union
import re

import torch
from PIL import Image
from transformers import PreTrainedTokenizer

from .constants import IMAGE_TOKEN

def create_mllama_message(user_text, image_token="<image>"):
    """
    Llama 3.2 Vision (Mllama)モデル用のメッセージを作成します。
    
    Args:
        user_text: ユーザーテキスト
        image_token: 画像トークン（デフォルトは<image>）
        
    Returns:
        メッセージ形式のリスト
    """
    # 画像トークンがない場合は先頭に追加
    if image_token not in user_text:
        user_text = f"{image_token} {user_text}"
    
    # Mllama用のメッセージ形式
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text}
            ]
        }
    ]
    return messages

def load_image_from_base64(image_str: str) -> Image.Image:
    """Base64エンコードされた画像文字列からPIL Imageをロードする"""
    image_data = base64.b64decode(image_str)
    return Image.open(BytesIO(image_data)).convert("RGB")

def tokenizer_image_token(
    prompt: str,
    tokenizer: PreTrainedTokenizer,
    return_tensors: Optional[str] = None
) -> Union[List[int], torch.Tensor]:
    """
    画像プレースホルダトークンを含む可能性のあるプロンプトをトークン化します。
    プレースホルダトークンは`<|image|>`です。
    この関数は、特殊な画像トークンがトークン化されたシーケンスに正しく挿入されるようにします。
    
    Args:
        prompt: プロンプト文字列。画像を挿入する箇所に`<|image|>`が含まれている場合があります。
        tokenizer: Llama3.2モデル用のトークナイザ。
        return_tensors: 'pt'の場合、PyTorchテンソルのトークンIDを返します。Noneの場合、トークンIDのリストを返します。
    """
    # 画像トークンに対応するパターン
    image_token_pattern = r"<image>"
    image_token_idx = tokenizer.convert_tokens_to_ids("<image>")
    
    # プロンプト内の画像トークンを置換
    parts = re.split(image_token_pattern, prompt)
    tokens = []
    
    for i, part in enumerate(parts):
        # パートをトークン化
        part_tokens = tokenizer.encode(part, add_special_tokens=False)
        tokens.extend(part_tokens)
        
        # 最後のパート以外では画像トークンを追加
        if i < len(parts) - 1:
            tokens.append(image_token_idx)
    
    # テンソル形式で返す
    if return_tensors == "pt":
        return torch.tensor([tokens])
    return tokens

def process_images(
    images: Union[Image.Image, List[Image.Image], str, List[str], BytesIO, List[BytesIO]],
    processor=None,
    image_processor=None,
    return_tensors='pt'
) -> Optional[torch.Tensor]:
    """
    1つまたは複数の画像をモデル入力用に処理します。
    モデルの画像プロセッサを使用して、画像をピクセル値テンソルに変換します。
    
    Args:
        images: PIL画像、またはPIL画像のリスト（またはPIL画像としてロードされる画像ファイルパス/base64文字列）
        processor: モデル用のtransformers.AutoProcessor（または類似品）のインスタンス（オプション）
        image_processor: 画像プロセッサ（例：MllamaImageProcessor）のインスタンス（オプション）
        return_tensors: 出力形式、PyTorchテンソルの場合は'pt'
    Returns:
        モデルに応じて(batch_size, num_image_tiles, channels, height, width)または(batch_size, channels, height, width)の
        形状のテンソル。プロセッサを使用する場合は通常['pixel_values']でアクセス可能。
    """
    if images is None:
        return None
        
    # 単一画像の場合はリストに変換
    if not isinstance(images, list):
        images = [images]
    
    # 画像を標準化されたPIL形式に変換
    new_images = []
    for image in images:
        if isinstance(image, str):
            image = load_image_from_base64(image)
        elif isinstance(image, BytesIO):
            image = Image.open(image)
        
        if not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # RGBに変換
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        new_images.append(image)
    
    # 適切なプロセッサで画像を処理
    if processor is not None:
        # MllamaProcessorなどの複合プロセッサの場合
        if hasattr(processor, 'image_processor'):
            outputs = processor.image_processor(new_images, return_tensors=return_tensors)
            if isinstance(outputs, dict) and "pixel_values" in outputs:
                return outputs["pixel_values"]
            return outputs
        return processor(new_images, return_tensors=return_tensors)
    
    # 画像プロセッサが指定されている場合
    if image_processor is not None:
        return image_processor(new_images, return_tensors=return_tensors)["pixel_values"]
    
    # どのプロセッサも指定されていない場合
    return None

class KeywordsStoppingCriteria(torch.nn.Module):
    """指定されたキーワードが出力に表示されたときにテキスト生成を停止する基準"""
    def __init__(self, keywords: List[str], tokenizer: PreTrainedTokenizer, initial_input_ids: torch.LongTensor):
        super().__init__()
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.initial_input_ids_len = initial_input_ids.shape[1]
        
        # キーワードごとにトークンIDリストを作成
        self.keyword_token_ids = []
        for keyword in keywords:
            keyword_tokens = tokenizer.encode(keyword, add_special_tokens=False)
            self.keyword_token_ids.append(torch.tensor(keyword_tokens))
        
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        # 生成された部分のみを考慮
        outputs = output_ids[:, self.initial_input_ids_len:]
        
        # バッチサイズは通常1
        for batch_idx in range(outputs.shape[0]):
            batch_output_ids = outputs[batch_idx]
            
            # 各キーワードについてチェック
            for keyword_ids in self.keyword_token_ids:
                # 出力シーケンス長がキーワードより短い場合はスキップ
                if len(batch_output_ids) < len(keyword_ids):
                    continue
                
                # 最後のn個（キーワード長）のトークンを取得
                last_n_ids = batch_output_ids[-(len(keyword_ids)):]
                
                # キーワードと一致するか確認
                if torch.all(last_n_ids == keyword_ids):
                    return True
        
        return False 