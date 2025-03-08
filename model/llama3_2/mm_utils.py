import base64
from io import BytesIO
from typing import List, Optional, Union, Dict, Any
import re

import torch
from PIL import Image
from transformers import PreTrainedTokenizer
import numpy as np
import requests
from transformers.modeling_outputs import BaseModelOutputWithPast

from .constants import IMAGE_TOKEN

def create_mllama_message(
    image: Union[str, Image.Image, BytesIO, None], 
    text: str, 
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Llama 3.2 Vision (Mllama)のプロセッサ用に適切なフォーマットのメッセージを作成します。
    Args:
        image: PIL Image、画像ファイルパス、BytesIO、またはBase64文字列
        text: プロンプト（テキスト部分）
        verbose: デバッグ情報を表示するかどうか
    Returns:
        プロセッサに渡すための辞書
    """
    # 画像が提供されていない場合は直接テキストを返す
    if image is None:
        if verbose:
            print("画像なし: テキストのみのメッセージを作成します")
        return {"text": text}
    
    # 画像の処理（文字列/パス、BytesIO、PIL.Imageに対応）
    processed_image = process_images(image)
    
    # MetaのMllamaProcessorが期待する形式
    # Processorのエラー回避のため、単純に画像とテキストをそのまま返す
    # これにより、tokenizer(**message)の呼び出しで適切に処理される
    if verbose:
        print(f"作成されたメッセージ: 画像とテキスト「{text[:30]}...」")
    
    # MllamaProcessorの期待する最もシンプルなフォーマット
    # 技術的には、画像はimagesキーに、テキストはtextキーに
    # このシンプルな形式はtokenizer(images=image, text=text)と同等
    return {
        "images": processed_image,
        "text": text
    }

def load_image_from_base64(image_str: str) -> Image.Image:
    """Base64エンコードされた画像文字列からPIL Imageをロードする"""
    image_data = base64.b64decode(image_str)
    return Image.open(BytesIO(image_data)).convert("RGB")

def tokenizer_image_token(text: str, image_token: str = "<image>") -> str:
    """
    テキスト内に画像トークンを挿入します（まだ存在しない場合）。
    
    Args:
        text: 元のテキスト
        image_token: 使用する画像トークン
    
    Returns:
        画像トークンが適切に挿入されたテキスト
    """
    if not text:
        # テキストが空または None の場合、画像トークンだけを返す
        return image_token
    
    # 画像トークンがすでに含まれている場合はそのまま返す
    if image_token in text:
        return text
    
    # 正規表現を使ってテキストを分解
    pattern = r"(.*?)(?:\s*$)"
    match = re.match(pattern, text)
    
    if match:
        # プロンプトの最後に画像トークンを追加
        return f"{match.group(1)} {image_token}"
    else:
        # マッチしない場合はそのまま画像トークンを追加
        return f"{text} {image_token}"

def process_images(images: Union[List[Union[str, Image.Image, BytesIO]], Union[str, Image.Image, BytesIO]]) -> Union[List[Image.Image], Image.Image]:
    """
    さまざまな形式の画像入力を処理し、PIL.Image形式に変換します。
    
    Args:
        images: 処理する画像（複数可）。以下の形式に対応:
            - PIL.Image: そのまま返す
            - 文字列: ファイルパス、URL、またはbase64文字列として解釈
            - BytesIO: PILで読み込む
            - リスト: 上記いずれかの形式の画像リスト

    Returns:
        PIL.Image形式の画像または画像リスト
    """
    if images is None:
        return None
    
    # 単一の画像を処理する場合
    if not isinstance(images, list):
        return process_single_image(images)
    
    # 画像のリストを処理する場合
    processed_images = []
    for image in images:
        processed_images.append(process_single_image(image))
    
    return processed_images

def process_single_image(image: Union[str, Image.Image, BytesIO]) -> Image.Image:
    """
    一つの画像を処理し、PIL.Image形式に変換します。
    
    Args:
        image: 処理する画像。以下の形式に対応:
            - PIL.Image: そのまま返す
            - 文字列: ファイルパス、URL、またはbase64文字列として解釈
            - BytesIO: PILで読み込む

    Returns:
        PIL.Image形式の画像
    """
    # すでにPIL.Image形式の場合
    if isinstance(image, Image.Image):
        return image
    
    # BytesIOオブジェクトの場合
    if isinstance(image, BytesIO):
        return Image.open(image).convert('RGB')
    
    # 文字列の場合、複数の可能性がある
    if isinstance(image, str):
        # URLの場合
        if image.startswith('http://') or image.startswith('https://'):
            try:
                response = requests.get(image, stream=True)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert('RGB')
            except Exception as e:
                print(f"URLからの画像取得に失敗: {e}")
                raise
        
        # Base64エンコードされた画像の場合
        if image.startswith('data:image'):
            return load_image_from_base64(image)
        
        # 通常のファイルパスと判断
        try:
            return Image.open(image).convert('RGB')
        except Exception as e:
            print(f"ファイルからの画像読み込みに失敗: {e}")
            # base64の可能性があるか試してみる
            try:
                return load_image_from_base64(image)
            except:
                # 全ての試行が失敗
                raise ValueError(f"画像の形式を解釈できません: {image[:30]}...")
    
    # サポートされていない形式
    raise ValueError(f"サポートされていない画像形式: {type(image)}")

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

def print_debug_info_for_mllama(tokenizer, model, inputs=None):
    """
    Mllamaモデルとそのトークナイザーに関するデバッグ情報を表示します。
    問題のトラブルシューティングに役立ちます。
    
    Args:
        tokenizer: MllamaProcessorなど、使用するトークナイザー/プロセッサ
        model: Llama3.2 Visionモデル
        inputs: オプションで、テスト入力を指定
    """
    print("\n===== Mllama Debug Info =====")
    
    # トークナイザー情報
    print("\nTokenizer/Processor Info:")
    print(f"Type: {type(tokenizer)}")
    print(f"Available methods: {[m for m in dir(tokenizer) if not m.startswith('_') and callable(getattr(tokenizer, m))[:10]]}")
    
    # トークナイザーに画像処理機能があるか
    has_image_processor = hasattr(tokenizer, 'image_processor')
    print(f"Has image_processor: {has_image_processor}")
    
    # モデル情報
    print("\nModel Info:")
    print(f"Type: {type(model)}")
    print(f"Config: {model.config.__class__.__name__}")
    
    # 入力情報
    if inputs is not None:
        print("\nInput Info:")
        print(f"Keys: {inputs.keys()}")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: Tensor of shape {v.shape}, dtype {v.dtype}, device {v.device}")
            else:
                print(f"  {k}: {type(v)}")
    
    print("===== End Debug Info =====\n")

def safe_mllama_process(tokenizer, images, text, max_tries=3, verbose=True):
    """
    Mllamaプロセッサを安全に使用するためのヘルパー関数。
    複数の処理方法を試し、最初に成功した方法を使用します。
    
    Args:
        tokenizer: Mllamaプロセッサ
        images: 処理する画像
        text: 処理するテキスト
        max_tries: 最大試行回数
        verbose: デバッグ出力を表示するかどうか
    
    Returns:
        処理された入力テンソル
    """
    if verbose:
        print(f"Mllamaプロセッサでの処理を試みます (試行回数: {max_tries})")
    
    # 画像の前処理
    processed_image = process_images(images)
    
    # 画像トークンの挿入
    text_with_token = tokenizer_image_token(text)
    
    errors = []
    
    # 方法1: 画像とテキストを直接渡す
    try:
        if verbose:
            print("方法1: 画像とテキストを直接渡す")
        inputs = tokenizer(processed_image, text_with_token, return_tensors="pt")
        if verbose:
            print("方法1で成功")
        return inputs
    except Exception as e:
        errors.append(f"方法1エラー: {str(e)}")
    
    # 方法2: 辞書形式で渡す
    try:
        if verbose:
            print("方法2: 辞書形式で渡す")
        inputs = tokenizer(images=processed_image, text=text_with_token, return_tensors="pt")
        if verbose:
            print("方法2で成功")
        return inputs
    except Exception as e:
        errors.append(f"方法2エラー: {str(e)}")
    
    # 方法3: create_mllama_message関数を使用
    try:
        if verbose:
            print("方法3: create_mllama_message関数を使用")
        message = create_mllama_message(processed_image, text, verbose=False)
        inputs = tokenizer(**message, return_tensors="pt")
        if verbose:
            print("方法3で成功")
        return inputs
    except Exception as e:
        errors.append(f"方法3エラー: {str(e)}")
    
    # 方法4: テキストのみで処理（フォールバック）
    try:
        if verbose:
            print("方法4: テキストのみで処理")
        inputs = tokenizer(text_with_token, return_tensors="pt")
        if verbose:
            print("方法4で成功 (ただし画像は処理されません)")
        return inputs
    except Exception as e:
        errors.append(f"方法4エラー: {str(e)}")
    
    # すべての方法が失敗した場合
    error_message = "\n".join(errors)
    raise RuntimeError(f"すべての処理方法が失敗しました:\n{error_message}")

def run_test_mllama_processor(processor, model_name="Llama-3.2-Vision"):
    """
    Mllamaプロセッサのテストを実行します。
    
    Args:
        processor: テストするMllamaプロセッサ
        model_name: モデル名（ログ表示用）
    
    Returns:
        テスト結果（成功した場合はTrue）
    """
    print(f"\n===== {model_name} プロセッサのテスト =====")
    
    # テスト用の画像（単色の小さな画像を作成）
    test_image = Image.new('RGB', (100, 100), color='blue')
    test_text = "Describe what you see in this image."
    
    try:
        # 基本的なプロセッサの呼び出し方法をテスト
        print("テスト1: 基本的な呼び出し（直接引数）")
        inputs = processor(test_image, test_text, return_tensors="pt")
        print("成功! 出力キー:", inputs.keys())
        
        # create_mllama_message関数のテスト
        print("\nテスト2: create_mllama_message関数の使用")
        message = create_mllama_message(test_image, test_text)
        inputs = processor(**message, return_tensors="pt")
        print("成功! 出力キー:", inputs.keys())
        
        # キーワード引数でのテスト
        print("\nテスト3: キーワード引数での呼び出し")
        inputs = processor(images=test_image, text=test_text, return_tensors="pt")
        print("成功! 出力キー:", inputs.keys())
        
        print(f"\n===== {model_name} プロセッサのテスト完了: すべて成功 =====")
        return True
        
    except Exception as e:
        print(f"テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n===== {model_name} プロセッサのテスト完了: 失敗 =====")
        return False 