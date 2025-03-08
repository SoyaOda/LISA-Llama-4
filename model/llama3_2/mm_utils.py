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
    
    # セグメンテーション要求の場合は特別な処理
    is_seg_request = "<SEG>" in text or "[SEG]" in text or "segment" in text.lower()
    
    # 明示的に画像トークンを追加（必要な場合）
    if "<|image|>" not in text:
        if is_seg_request:
            # セグメンテーション要求の場合、先頭に画像トークンを追加
            text = "<|image|> " + text
        else:
            # 一般的な場合
            text = "<|image|> " + text
    
    if verbose:
        print(f"作成されたMllamaメッセージ: {text[:50]}...")
    
    # MllamaProcessorの期待する形式
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
    テキスト中で画像トークンがない場合、適切に追加する関数
    Args:
        text: プロンプトテキスト
        image_token: 使用する画像トークン
    Returns:
        画像トークン付きのプロンプト
    """
    # すでに画像トークンがある場合はそのまま返す
    if image_token in text:
        return text
    
    # これはLlama 3.2 Vision (Mllama)では必須ではないが、
    # 必要に応じて画像トークンを追加する場合のロジック
    
    # 既に<|image|>が含まれていればそのまま返す
    if "<|image|>" in text:
        return text
        
    # セグメンテーションコマンドの場合、できるだけ前に挿入
    if "segment" in text.lower() or "<SEG>" in text or "[SEG]" in text:
        # セグメント指示がある場合、最初の文の前に画像トークンを挿入
        sentences = text.split(". ")
        if len(sentences) > 1:
            # 最初の文の後に画像トークンを挿入
            return sentences[0] + ". " + image_token + " " + ". ".join(sentences[1:])
        else:
            # 一文しかない場合は文頭に挿入
            return image_token + " " + text
    
    # 一般的なケース
    # Mllama用の特別なプロンプト形式の試行
    if not text.startswith(image_token):
        return image_token + " " + text
    
    return text

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

def process_mllama_pixel_values(pixel_values, target_format="sam", verbose=True):
    """
    Mllamaプロセッサのpixel_values出力を別の形式に変換します
    Args:
        pixel_values: MllamaProcessorから出力されたpixel_values
        target_format: 'sam'など、目的の形式
        verbose: デバッグ情報を表示するかどうか
    Returns:
        変換されたpixel_values
    """
    if pixel_values is None:
        return None
    
    original_shape = pixel_values.shape
    if verbose:
        print(f"元のpixel_values形状: {original_shape}, 型: {pixel_values.dtype}")
    
    # Mllamaプロセッサの出力形状を処理
    if target_format == "sam":
        # SAMは[バッチ, チャンネル, 高さ, 幅]形式を期待
        if len(original_shape) == 4:
            # すでに正しい形状
            # ピクセル値の範囲を確認
            if verbose:
                min_val = pixel_values.min().item()
                max_val = pixel_values.max().item()
                print(f"ピクセル値の範囲: [{min_val}, {max_val}]")
            
            return pixel_values
        elif len(original_shape) == 6 and original_shape[0] == 1 and original_shape[1] == 1:
            # 形状[1, 1, 4, 3, 560, 560]など - MLlamaProcessorからの出力
            # 最初の有用な画像データを使用（最初の画像）
            if verbose:
                print("複雑な形状を[バッチ, チャンネル, 高さ, 幅]形式に変換します")
            
            # [0, 0, 0]で最初の画像を取得し、バッチ次元を追加
            # MllamaProcessorの出力は通常 [バッチ, seq_len, img_token_pos, チャンネル, 高さ, 幅]
            # 元のLISAコードと一致するように、最初の画像トークンを使用
            converted = pixel_values[0, 0, 0].unsqueeze(0)
            
            if verbose:
                min_val = converted.min().item()
                max_val = converted.max().item()
                print(f"変換後の形状: {converted.shape}, 範囲: [{min_val}, {max_val}]")
            
            return converted
        elif len(original_shape) == 5:
            # 形状[バッチ, トークン位置, チャンネル, 高さ, 幅]の場合
            if verbose:
                print(f"5次元形状を処理: {original_shape}")
            
            # 最初の画像トークンを取得
            converted = pixel_values[0, 0].unsqueeze(0)
            
            if verbose:
                min_val = converted.min().item()
                max_val = converted.max().item()
                print(f"変換後の形状: {converted.shape}, 範囲: [{min_val}, {max_val}]")
            
            return converted
        else:
            # その他の形状の場合は可能な限り変換を試みる
            if verbose:
                print(f"警告: 予期しない形状: {original_shape}, 変換を試みます")
            
            # バッチ次元を保持し、チャンネル、高さ、幅の次元を検出する
            if len(original_shape) > 4:
                # 最終の3次元をチャンネル、高さ、幅と見なす
                try:
                    # 最初のバッチだけ使用
                    reshaped = pixel_values.view(1, -1, original_shape[-3], original_shape[-2], original_shape[-1])
                    # 最初の有効な画像トークンを使用
                    converted = reshaped[0, 0].unsqueeze(0)
                    
                    if verbose:
                        min_val = converted.min().item()
                        max_val = converted.max().item()
                        print(f"変換後の形状: {converted.shape}, 範囲: [{min_val}, {max_val}]")
                    
                    return converted
                except Exception as e:
                    if verbose:
                        print(f"リシェイプエラー: {e}")
                    # ベストエフォートで変換を試みる
                    pass
            
            # もし上記の変換が失敗した場合、元の形状を返す
            if verbose:
                print("警告: 適切な変換が見つかりませんでした。元の形状を返します")
            return pixel_values
    else:
        if verbose:
            print(f"未知のターゲット形式: {target_format}, 元の形状を返します")
        return pixel_values


def is_valid_sam_input(tensor):
    """
    テンソルがSAM用の入力として有効かどうかをチェックします
    Args:
        tensor: 確認するテンソル
    Returns:
        bool: 有効な場合はTrue
    """
    if tensor is None:
        return False
    
    # 4次元テンソル([バッチ, チャンネル, 高さ, 幅])であることを確認
    if len(tensor.shape) != 4:
        return False
    
    # チャンネル数が3であることを確認（RGB）
    if tensor.shape[1] != 3:
        return False
    
    return True


def safe_mllama_to_sam(pixel_values, target_size=1024):
    """
    MllamaのPIXEL_VALUESをSAM用に安全に変換する便利関数
    Args:
        pixel_values: MllamaProcessorからのpixel_values
        target_size: 変換後の目標サイズ (SAMは通常1024x1024を想定)
    Returns:
        SAM用に適切に変換されたテンソル
    """
    import torch
    import numpy as np
    
    # すでに適切な形状ならそのまま返す
    if is_valid_sam_input(pixel_values):
        # サイズをチェック - SAMは大きなサイズを想定
        h, w = pixel_values.shape[2], pixel_values.shape[3]
        if h >= target_size and w >= target_size:
            return pixel_values
    
    # 変換を試みる
    converted = process_mllama_pixel_values(pixel_values, target_format="sam")
    
    # 変換後も無効なら例外
    if not is_valid_sam_input(converted):
        raise ValueError(f"有効なSAM入力に変換できません。形状: {converted.shape}")
    
    # サイズをチェック
    h, w = converted.shape[2], converted.shape[3]
    if h < target_size or w < target_size:
        print(f"警告: 変換後の画像サイズ ({h}x{w}) がSAMの想定より小さいです。")
        print(f"SAMはより大きなサイズ (約{target_size}x{target_size}) を想定しています。")
        
        try:
            # リサイズが必要な場合はNumPy経由で変換
            from model.segment_anything.utils.transforms import ResizeLongestSide
            
            # 画像をNumPy配列に変換
            device = converted.device
            img_np = converted[0].permute(1, 2, 0).cpu().numpy()
            
            # NumPy配列をuint8形式に変換（SAMのtransformsが期待する形式）
            if img_np.dtype == np.float32:
                # [0,1]の範囲にある場合は255を掛ける
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255.0).astype(np.uint8)
                else:
                    # すでに[0,255]の範囲にある場合はそのままuint8に変換
                    img_np = img_np.astype(np.uint8)
            
            print(f"リサイズ前のNumPy配列型: {img_np.dtype}, 範囲: [{img_np.min()}, {img_np.max()}]")
            
            # SAMのResizeLongestSideを使用してリサイズ
            transform = ResizeLongestSide(target_size)
            resized_img = transform.apply_image(img_np)
            print(f"リサイズ後の形状: {resized_img.shape}")
            
            # 前処理関数（SAMが期待する正規化と形状に変換）
            import torch.nn.functional as F
            def preprocess(x, pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
                        pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), img_size=1024):
                # Normalize colors
                x = (x - pixel_mean) / pixel_std
                # Pad
                h, w = x.shape[-2:]
                padh = img_size - h
                padw = img_size - w
                x = F.pad(x, (0, padw, 0, padh))
                return x
            
            # テンソルに変換して前処理
            image_tensor = torch.from_numpy(resized_img).permute(2, 0, 1).contiguous()
            image_tensor = preprocess(image_tensor).unsqueeze(0).to(device)
            print(f"前処理後のテンソル形状: {image_tensor.shape}")
            
            # 精度合わせ
            if converted.dtype == torch.float16:
                image_tensor = image_tensor.half()
            elif converted.dtype == torch.bfloat16:
                image_tensor = image_tensor.bfloat16()
            
            return image_tensor
            
        except Exception as e:
            print(f"リサイズ中にエラーが発生: {e}")
            import traceback
            traceback.print_exc()
            # エラーが発生した場合は変換前のテンソルを返す
            print("元のテンソルをそのまま返します")
    
    return converted 