import base64
from io import BytesIO
from typing import List, Optional, Union

import torch
from PIL import Image
from transformers import PreTrainedTokenizer

from .constants import IMAGE_TOKEN

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
    # プロンプトを画像プレースホルダトークンで分割
    parts = prompt.split(IMAGE_TOKEN)
    tokenized_chunks = []
    
    for part in parts:
        if part == "":
            token_ids = []  # このチャンクにはテキストがない
        else:
            # 特殊トークンを追加せずにテキストチャンクをトークン化（BOSを繰り返し追加しないため）
            tokens = tokenizer(part, add_special_tokens=False)
            token_ids = tokens["input_ids"] if isinstance(tokens, dict) else tokens
        tokenized_chunks.append(token_ids if isinstance(token_ids, list) else list(token_ids))
    
    # 画像プレースホルダのトークンIDを取得
    try:
        image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    except Exception as e:
        raise ValueError(f"トークナイザには{IMAGE_TOKEN}トークンが定義されていません。") from e

    # トークナイザがBOSトークンを使用する場合、シーケンスの先頭に追加（一度だけ）
    bos_id = tokenizer.bos_token_id
    output_ids: List[int] = []
    if bos_id is not None:
        output_ids.append(bos_id)
    
    # テキストチャンクと画像トークンIDを交互に配置
    for i, chunk in enumerate(tokenized_chunks):
        # トークン化されたテキストチャンクを追加
        output_ids.extend(chunk)
        # 各チャンク（最後を除く）の後に画像トークンを追加
        if i < len(tokenized_chunks) - 1:
            output_ids.append(image_token_id)
    
    # 要求があれば、テンソルに変換
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor([output_ids], dtype=torch.long)  # shape (1, sequence_length)
        else:
            raise ValueError(f"サポートされていないreturn_tensorsタイプ: {return_tensors}")
    return output_ids

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
    # PIL画像のリストを確保
    pil_images = []
    if images is None:
        return None
    
    if isinstance(images, list):
        img_list = images
    else:
        img_list = [images]
        
    for img in img_list:
        if isinstance(img, str):
            # 画像が文字列の場合、ファイルパスまたはURLと仮定
            # （注：URLの場合、ユーザーは外部で処理する必要があります。ここではローカルパスを処理します）
            try:
                # ファイルパスとして開く
                pil_img = Image.open(img).convert("RGB")
            except Exception as e:
                # 失敗した場合、base64文字列として扱う
                pil_img = load_image_from_base64(img)
        elif isinstance(img, BytesIO):
            pil_img = Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image):
            pil_img = img.convert("RGB")
        else:
            raise ValueError("サポートされていない画像形式です。PIL Images、ファイルパス、またはbase64文字列を提供してください。")
        pil_images.append(pil_img)
    
    # 提供されたプロセッサまたは画像プロセッサを使用してピクセル値を取得
    if processor is not None:
        # プロセッサはトークナイザと画像プロセッサを組み合わせたもの
        processed = processor(images=pil_images, return_tensors=return_tensors)
        # processedはBatchEncodingまたは類似のもので、'pixel_values'を含む
        return processed.get("pixel_values", None)
    elif image_processor is not None:
        processed = image_processor(images=pil_images, return_tensors=return_tensors)
        return processed.get("pixel_values", None)
    else:
        raise ValueError("画像を処理するには、プロセッサまたは画像プロセッサを提供する必要があります。")

class KeywordsStoppingCriteria(torch.nn.Module):
    """指定されたキーワードが出力に表示されたときにテキスト生成を停止する基準"""
    def __init__(self, keywords: List[str], tokenizer: PreTrainedTokenizer, initial_input_ids: torch.LongTensor):
        super().__init__()
        self.keywords = keywords
        # 各キーワードのトークンIDを事前に計算
        self.keyword_ids = []
        self.max_keyword_len = 0
        for kw in keywords:
            kw_ids = tokenizer(kw, add_special_tokens=False)["input_ids"]
            if len(kw_ids) == 0:
                continue
            # 先頭にBOSがあれば削除
            if kw_ids[0] == tokenizer.bos_token_id:
                kw_ids = kw_ids[1:]
            self.max_keyword_len = max(self.max_keyword_len, len(kw_ids))
            self.keyword_ids.append(torch.tensor(kw_ids, dtype=torch.long))
        self.start_len = initial_input_ids.shape[-1]  # 入力プロンプトの長さ（出力チェックで無視するため）
    
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        # 生成された部分のみを考慮
        gen_sequence = output_ids[0, self.start_len:]  # 最初のシーケンスの新しく生成されたトークンのみを取得
        if gen_sequence.numel() == 0:
            return False
        for kw_id_seq in self.keyword_ids:
            seq_len = kw_id_seq.shape[0]
            if seq_len > gen_sequence.shape[0]:
                continue
            # 生成されたシーケンスの末尾部分がキーワードシーケンスと一致するかチェック
            if torch.equal(gen_sequence[-seq_len:], kw_id_seq.to(gen_sequence.device)):
                return True
        return False 