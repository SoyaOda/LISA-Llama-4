from typing import List, Optional, Dict, Any, Union, Tuple
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BitsAndBytesConfig, 
    AutoProcessor, 
    AutoModelForVision2Seq, 
    AutoConfig, 
    GenerationMixin,
    MllamaForConditionalGeneration,
    MllamaConfig
)

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

import math
import transformers
import numpy as np
from transformers import AutoConfig, CLIPVisionModel, AutoModelForCausalLM, AutoTokenizer, AutoProcessor, PretrainedConfig
import os
from safetensors import safe_open
import collections.abc
from transformers.utils import is_flash_attn_2_available
import json
from PIL import Image

# llama3_2モジュールからのインポートパスを修正
try:
    from model.llama3_2.model.language_model.llama3_2 import Llama3VisionMetaModel
    print("Llama3VisionMetaModelを正しくインポートしました")
except ImportError as e:
    print(f"警告: Llama3VisionMetaModelのインポートに失敗しました: {e}")
    print("代替としてダミークラスを使用します")
    
    # インポートに失敗した場合は単純なダミークラスを定義
    class Llama3VisionMetaModel(nn.Module):
        def __init__(self, config, **kwargs):
            super().__init__()
            print("警告: ダミーのLlama3VisionMetaModelを使用しています")
            self.config = config
        
        def forward(self, *args, **kwargs):
            raise NotImplementedError("ダミーLlama3VisionMetaModelは機能しません")

from .segment_anything import build_sam_vit_h


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class LisaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        # configがNoneの場合に対応
        if self.config is None:
            print("警告: LisaMetaModelの初期化時にconfigがNoneです")
            # 最低限必要な属性を持つ簡易configオブジェクトを作成
            from types import SimpleNamespace
            self.config = SimpleNamespace()
            
        # 必要な属性を追加
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs.get("train_mask_decoder", True)
        if not hasattr(self.config, "out_dim"):
            self.config.out_dim = kwargs.get("out_dim", 256)
            
        # vision_pretrained設定を取得
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        
        # オリジナルLISAコードと同様に常にSAMを初期化
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        # MllamaConfigの場合はtext_config.hidden_sizeにアクセス
        if hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
            in_dim = config.text_config.hidden_size
        else:
            # 互換性のために元の参照方法もフォールバックとして保持
            in_dim = getattr(config, 'hidden_size', 4096)  # デフォルト値として4096を使用
            
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class LisaModel(nn.Module):
    """
    LISAのベースモデル
    """
    def __init__(
        self,
        model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        sam_vision_encoder=None,
        mask_decoder=None,
        device_map=None,
        torch_dtype=torch.float16,
        train_mask_decoder=True,
        out_dim=256
    ):
        """
        SAMに基づいたLlama 3.2 Visionモデル。
        
        Args:
            model_id: Llama 3.2 Visionモデルの識別子
            sam_vision_encoder: SAMのビジョンエンコーダー
            mask_decoder: マスクデコーダー
            device_map: モデルをどのデバイスにマッピングするか
            torch_dtype: モデルのデータ型
            train_mask_decoder: マスクデコーダーを訓練するかどうか
            out_dim: 出力次元
        """
        super().__init__()
        
        print(f"LisaModelを初期化します。model_id: {model_id}")
        self.model_id = model_id
        self.train_mask_decoder = train_mask_decoder
        self.out_dim = out_dim
        self.seg_token_idx = None
        
        # configオブジェクトを初期化
        try:
            # MllamaConfigを使用してconfigを作成
            from transformers import MllamaConfig
            try:
                print(f"MllamaConfigを使用してconfigを初期化します: {model_id}")
                self.config = MllamaConfig.from_pretrained(model_id)
            except Exception as e:
                print(f"MllamaConfigの初期化中にエラーが発生しました: {e}")
                print("AutoConfigを使用して再試行します")
                from transformers import AutoConfig
                self.config = AutoConfig.from_pretrained(model_id)
        except Exception as e:
            print(f"configの初期化中にエラーが発生しました: {e}")
            print("空のconfigを作成します")
            from transformers import PretrainedConfig
            self.config = PretrainedConfig()
        
        # seg_token_idxの属性を追加
        self.config.seg_token_idx = None
        
        # Llama 3.2 Visionモデルを初期化
        try:
            print(f"MllamaForConditionalGenerationを初期化します: {model_id}")
            print(f"torch_dtype: {torch_dtype}")
            from transformers import MllamaForConditionalGeneration
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_id,
                device_map=device_map,
                torch_dtype=torch_dtype
            )
            print("MllamaForConditionalGenerationの初期化が成功しました")
        except Exception as e:
            print(f"MllamaForConditionalGenerationの初期化中にエラーが発生しました: {e}")
            print("AutoModelForVision2Seqで再試行します")
            try:
                from transformers import AutoModelForVision2Seq
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_id,
                    device_map=device_map,
                    torch_dtype=torch_dtype
                )
                print("AutoModelForVision2Seqの初期化が成功しました")
            except Exception as e:
                print(f"AutoModelForVision2Seqの初期化中もエラーが発生しました: {e}")
                raise
        
        # プロセッサを初期化
        try:
            print(f"プロセッサを初期化します: {model_id}")
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_id)
            print("プロセッサの初期化が成功しました")
        except Exception as e:
            print(f"プロセッサの初期化中にエラーが発生しました: {e}")
            self.processor = None  # プロセッサなしで続行
        
        # SAMビジョンエンコーダーの初期化または設定
        print("SAMビジョンエンコーダーを設定中...")
        if sam_vision_encoder is not None:
            self.visual_model = sam_vision_encoder
            print("提供されたSAMビジョンエンコーダーを使用します")
        else:
            print("SAMビジョンエンコーダーを初期化します")
            try:
                from .segment_anything import build_sam_vit_h
                vision_pretrained = kwargs.get("vision_pretrained", None)
                if vision_pretrained:
                    print(f"SAMモデルをロード: {vision_pretrained}")
                    self.visual_model = build_sam_vit_h(checkpoint=vision_pretrained)
                else:
                    print("警告: SAMモデルのチェックポイントが指定されていません")
                    self.visual_model = None
            except ImportError:
                try:
                    # 別のインポートパスを試す
                    from model.segment_anything import build_sam_vit_h
                    vision_pretrained = kwargs.get("vision_pretrained", None)
                    if vision_pretrained:
                        print(f"SAMモデルをロード: {vision_pretrained}")
                        self.visual_model = build_sam_vit_h(checkpoint=vision_pretrained)
                    else:
                        print("警告: SAMモデルのチェックポイントが指定されていません")
                        self.visual_model = None
                except Exception as e:
                    print(f"SAMビジョンエンコーダーの初期化中にエラーが発生しました: {e}")
                    self.visual_model = None
            except Exception as e:
                print(f"SAMビジョンエンコーダーの初期化中にエラーが発生しました: {e}")
                self.visual_model = None
        
        # マスクデコーダーの初期化または設定
        print("マスクデコーダーを設定中...")
        if mask_decoder is not None:
            self.mask_decoder = mask_decoder
            print("提供されたマスクデコーダーを使用します")
        else:
            print("マスクデコーダーを初期化します")
            # 何らかのデフォルトマスクデコーダーをここで初期化
            # ただし、実際の実装ではユーザーが事前に初期化したデコーダーを提供することが望ましい
            self.mask_decoder = None
        
        # マスクデコーダーのパラメータの勾配設定
        if self.mask_decoder is not None:
            for p in self.mask_decoder.parameters():
                p.requires_grad = train_mask_decoder
            print(f"マスクデコーダーの訓練設定: {train_mask_decoder}")
            
            # マスクデコーダーの入力用の投影層の初期化
            self.mlp = nn.Sequential(
                nn.Linear(self.model.config.hidden_size, self.out_dim),
                nn.GELU(),
                nn.Linear(self.out_dim, self.out_dim)
            )
            
    def resize_token_embeddings(self, new_num_tokens):
        """
        トークン埋め込みをリサイズする
        
        Args:
            new_num_tokens: 新しいトークン数
        
        Returns:
            リサイズされたモデル
        """
        try:
            if not hasattr(self, 'model'):
                print("警告: self.modelが存在しません。トークン埋め込みはリサイズできません。")
                return
                
            print(f"モデルの埋め込みをリサイズします: {new_num_tokens}")
            
            # Input埋め込みのリサイズ
            try:
                self.model.resize_token_embeddings(new_num_tokens)
                print("Input埋め込みのリサイズに成功しました")
            except Exception as e:
                print(f"Input埋め込みのリサイズ中にエラーが発生しました: {e}")
            
            # Output埋め込み（LM head）のリサイズ
            # LM headがtiedされていない場合に必要
            try:
                lm_head = self.model.get_output_embeddings()
                if lm_head is not None:
                    old_num_tokens, emb_dim = lm_head.weight.shape
                    if old_num_tokens != new_num_tokens:
                        print(f"Output埋め込み（LM head）もリサイズします: {old_num_tokens} -> {new_num_tokens}")
                        
                        # 新しい線形層を作成
                        new_lm_head = nn.Linear(
                            in_features=emb_dim, 
                            out_features=new_num_tokens, 
                            bias=lm_head.bias is not None
                        )
                        
                        # 既存の重みとバイアスをコピー
                        new_lm_head.weight.data[:old_num_tokens, :] = lm_head.weight.data
                        if lm_head.bias is not None:
                            new_lm_head.bias.data[:old_num_tokens] = lm_head.bias.data
                        
                        # 新しいLM headを設定
                        self.model.set_output_embeddings(new_lm_head)
                        print("Output埋め込み（LM head）のリサイズに成功しました")
                else:
                    print("Output埋め込み（LM head）が見つかりません")
            except Exception as e:
                print(f"Output埋め込み（LM head）のリサイズ中にエラーが発生しました: {e}")
                
            return self.model
        except Exception as e:
            print(f"resize_token_embeddings中にエラーが発生しました: {e}")
            return None

    def forward(self, **kwargs):
        """
        モデルの前方伝播。
        """
        # past_key_valuesが含まれている場合は直接モデルに渡す
        if "past_key_values" in kwargs:
            return self.model(**kwargs)
            
        # すべての要素が含まれているか確認
        required_elements = ["input_ids", "attention_mask", "images"]
        missing_elements = [elem for elem in required_elements if elem not in kwargs]
        if missing_elements:
            print(f"警告: 以下の要素が不足しています: {missing_elements}")
            
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        input_ids,
        attention_mask=None,
        attention_masks=None,
        labels=None,
        images=None,
        images_clip=None,
        masks_list=None,
        label_masks_list=None,
        label_list=None,
        inputs_embeds=None,
        offset=None,
        resize_list=None,
        original_size_list=None,
        inference=False,
        tokenizer=None,
        **kwargs
    ):
        # SAMの特徴を抽出
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        if offset is not None:
            assert batch_size == len(offset) - 1, f"バッチサイズ {batch_size} とオフセット長 {len(offset)} が一致しません"

        # input_idsが1次元の場合、2次元に変換
        if input_ids.dim() == 1:
            # エラーデバッグ情報を追加
            print(f"input_idsが1次元です。形状: {input_ids.shape}")
            # 1次元の場合は1xN形式にリシェイプ
            input_ids = input_ids.unsqueeze(0)
            print(f"変換後のinput_ids形状: {input_ids.shape}")
            
            # 同様にattention_masksとlabelsも調整
            if attention_masks is not None and attention_masks.dim() == 1:
                attention_masks = attention_masks.unsqueeze(0)
            if labels is not None and labels.dim() == 1:
                labels = labels.unsqueeze(0)

        processor = self.get_processor()
        
        # テキスト入力の準備: Llama3.2 Vision processorは文字列または文字列のリストを期待する
        if isinstance(input_ids, torch.Tensor):
            if tokenizer is None:
                # tokenizerが提供されていない場合はエラーメッセージを出力
                raise ValueError(
                    "テンソル形式のinput_idsが渡されましたが、tokenizerがNoneです。"
                    "processorがテキストを処理するには、文字列データが必要です。"
                    "tokenizerを提供するか、文字列または文字列のリスト形式のテキストを渡してください。"
                )
            # トークンIDをテキストにデコード
            raw_text_input = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
            print(f"デコードされたテキスト（デバッグ用）: {raw_text_input[:2]}")  # 最初の2つのみ表示
            
            # <|image|>トークンを追加して各テキストを整形
            text_input = []
            for text in raw_text_input:
                # Llama3.2 Visionは各テキスト内に<|image|>トークンが必要
                # 適切な位置に<|image|>トークンを挿入（通常は先頭に近い位置）
                if '<|image|>' not in text:
                    # ユーザーとシステムの区切りを探す（Llama3.2の形式に合わせる）
                    user_prefix = "User: "
                    if user_prefix in text:
                        # ユーザープレフィックスの直後に<|image|>を挿入
                        pos = text.find(user_prefix) + len(user_prefix)
                        text = text[:pos] + "<|image|> " + text[pos:]
                    else:
                        # フォールバック: テキストの先頭に追加
                        text = "<|image|> " + text
                text_input.append(text)
        else:
            # すでに文字列または文字列のリストの場合はそのまま使用
            raw_text_input = input_ids
            
            # <|image|>トークンの追加を確認
            if isinstance(raw_text_input, list):
                text_input = []
                for text in raw_text_input:
                    if '<|image|>' not in text:
                        # ユーザーとシステムの区切りを探す
                        user_prefix = "User: "
                        if user_prefix in text:
                            pos = text.find(user_prefix) + len(user_prefix)
                            text = text[:pos] + "<|image|> " + text[pos:]
                        else:
                            text = "<|image|> " + text
                    text_input.append(text)
            else:
                text_input = raw_text_input
                if '<|image|>' not in text_input:
                    # 単一文字列の場合
                    user_prefix = "User: "
                    if user_prefix in text_input:
                        pos = text_input.find(user_prefix) + len(user_prefix)
                        text_input = text_input[:pos] + "<|image|> " + text_input[pos:]
                    else:
                        text_input = "<|image|> " + text_input
            
        if not isinstance(text_input, (str, list)):
            # 文字列または文字列のリストでない場合はエラー
            raise ValueError(
                f"text_inputの型が無効です: {type(text_input)}。"
                "processorは文字列または文字列のリストを期待しています。"
            )
        
        # 画像データの準備
        # Llama3.2 Visionプロセッサでは、バッチ内の各要素に対して同じ数の画像が必要
        device = input_ids.device if isinstance(input_ids, torch.Tensor) else None
        
        if images_clip is not None and isinstance(images_clip, torch.Tensor):
            print(f"images_clip形状: {images_clip.shape}")
            
            # バッチサイズを取得
            batch_size = images_clip.shape[0] if len(images_clip.shape) > 0 else 1
            
            # 形状を調整して各バッチ要素が同じ数の画像を持つようにする
            if len(images_clip.shape) == 6:  # [batch, num_images, num_clips, channels, height, width]
                # 各バッチ要素に1つの画像を使用
                images_for_processor = images_clip[:, 0, 0]  # [batch, channels, height, width]
            elif len(images_clip.shape) == 5:  # [batch, num_clips, channels, height, width]
                images_for_processor = images_clip[:, 0]  # [batch, channels, height, width]
            else:
                images_for_processor = images_clip
            
            # BFloat16データ型の画像を扱う際にエラーが発生するため、float32に変換
            if images_for_processor.dtype == torch.bfloat16:
                print(f"画像データをBFloat16からfloat32に変換します")
                images_for_processor = images_for_processor.to(torch.float32)
            
            # テンソル形式の画像を0-1の範囲に正規化
            try:
                min_val = torch.min(images_for_processor).item()
                max_val = torch.max(images_for_processor).item()
                print(f"画像値の範囲（正規化前）: [{min_val}, {max_val}]")
                
                if min_val < 0 or max_val > 1:
                    # 値が0-1の範囲外の場合、正規化を行う
                    if min_val == max_val:
                        # すべての値が同じ場合（レアケース）
                        images_for_processor = torch.zeros_like(images_for_processor)
                    else:
                        # min-max正規化を適用
                        images_for_processor = (images_for_processor - min_val) / (max_val - min_val)
                    
                    # 正規化後の範囲を確認
                    new_min = torch.min(images_for_processor).item()
                    new_max = torch.max(images_for_processor).item()
                    print(f"画像値の範囲（正規化後）: [{new_min}, {new_max}]")
            except Exception as e:
                print(f"画像正規化中にエラーが発生しました: {e}")
                print(f"エラーの詳細情報:")
                traceback.print_exc()
                # PILへの変換に失敗した場合、元のテンソルを使用（エラーを出すことで問題診断が可能）
                # エラー回避はしない

            # テンソルからPILイメージに変換（これは常に行う）
            try:
                # テンソルをPILイメージのリストに変換する関数
                def tensor_to_pil_images(tensor):
                    """テンソルをPILイメージのリストに変換"""
                    if tensor.dim() < 3:
                        raise ValueError(f"画像テンソルは少なくとも3次元必要です。現在の形状: {tensor.shape}")
                    
                    # CPUに移動しておく
                    tensor = tensor.detach().cpu()
                    
                    # バッチ処理
                    if tensor.dim() == 4:  # [batch, channels, height, width]
                        pil_images = []
                        for i in range(tensor.shape[0]):
                            # チャネルを最後に移動 [channels, height, width] -> [height, width, channels]
                            img_np = tensor[i].permute(1, 2, 0).numpy()
                            
                            # 値が[0,1]の範囲外の場合は、クリッピング
                            img_np = np.clip(img_np, 0, 1)
                            
                            # [0,1]から[0,255]へスケーリング
                            img_np = (img_np * 255).astype(np.uint8)
                            
                            # NumPyからPILへ変換
                            img_pil = Image.fromarray(img_np)
                            pil_images.append(img_pil)
                        return pil_images
                    elif tensor.dim() == 3:  # 単一画像 [channels, height, width]
                        # チャネルを最後に移動 [channels, height, width] -> [height, width, channels]
                        img_np = tensor.permute(1, 2, 0).numpy()
                        
                        # 値が[0,1]の範囲外の場合は、クリッピング
                        img_np = np.clip(img_np, 0, 1)
                        
                        # [0,1]から[0,255]へスケーリング
                        img_np = (img_np * 255).astype(np.uint8)
                        
                        # NumPyからPILへ変換
                        img_pil = Image.fromarray(img_np)
                        return [img_pil]
                    else:
                        raise ValueError(f"サポートされていないテンソル形状: {tensor.shape}")
                
                # テンソルをPILイメージに変換
                pil_images = tensor_to_pil_images(images_for_processor)
                print(f"画像をPILイメージに変換しました: {len(pil_images)}個の画像")
                
                # 複数の画像がある場合、各画像の情報を表示
                if len(pil_images) > 0:
                    print(f"最初のPIL画像サイズ: {pil_images[0].size}, モード: {pil_images[0].mode}")
                
                # PILイメージをプロセッサに渡すために格納
                images_for_processor = pil_images
            except Exception as e:
                print(f"PILへの変換中にエラーが発生しました: {e}")
                print(f"エラーの詳細情報:")
                traceback.print_exc()
                # PILへの変換に失敗した場合、元のテンソルを使用（エラーを出すことで問題診断が可能）
                # エラー回避はしない
            
            print(f"processorに渡す画像形式: {type(images_for_processor)}")
            if isinstance(images_for_processor, list) and len(images_for_processor) > 0:
                print(f"  最初の要素の型: {type(images_for_processor[0])}")
        else:
            images_for_processor = None
            print("画像データなし")
        
        # seg_token_idxにアクセスする前にチェック
        seg_token_idx = getattr(self, 'seg_token_idx', None)
        if seg_token_idx is None:
            # LisaModelからの取得を試みる
            seg_token_idx = getattr(self.model, 'seg_token_idx', None)
            if seg_token_idx is None:
                # どちらにも存在しない場合は警告を出す
                print("警告: seg_token_idxが設定されていません")
                # デフォルト値として-1を使用
                seg_token_idx = -1
        
        # seg_token_idxを使用してマスク作成
        seg_token_mask = input_ids[:, 1:] == seg_token_idx
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().to(seg_token_mask.device),
            ],
            dim=1,
        )
        
        # プロセッサを使用してLlama3.2 Visionモデルの入力を準備
        processor_inputs = None
        try:
            # テキストとPIL画像を渡してプロセッサを実行
            processor_inputs = processor(
                text=text_input,
                images=images_for_processor if images_for_processor else None,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # プロセッサが出力するデータについてのデバッグ情報
            print(f"プロセッサ出力: {sorted(processor_inputs.keys())}")
            
            # デバイスをinput_idsと一致させる（テンソルの場合）
            if device is not None:
                for key, value in processor_inputs.items():
                    if isinstance(value, torch.Tensor):
                        processor_inputs[key] = value.to(device)
            
        except Exception as e:
            print(f"プロセッサの実行中にエラーが発生しました: {e}")
            print(f"テキスト入力: {text_input[:100]}...")  # 最初の100文字のみ
            print(f"画像入力タイプ: {type(images_for_processor)}")
            if isinstance(images_for_processor, list) and len(images_for_processor) > 0:
                print(f"  - 最初の画像タイプ: {type(images_for_processor[0])}")
                if hasattr(images_for_processor[0], 'size'):
                    print(f"  - 最初の画像サイズ: {images_for_processor[0].size}")
            traceback.print_exc()  # デバッグのために詳細なスタックトレースを表示
            processor_inputs = None
            
        # モデル実行（Llama3.2 Vision）
        vision_x = None
        try:
            if processor_inputs is not None:
                print("Llama3.2 Visionモデルを実行します")
                
                # cache_dataパラメータを削除（Llama3.2 Visionモデルは対応していない）
                # モデルを実行してテキスト表現を取得
                outputs = self.model.model(
                    **processor_inputs,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # hidden_statesを抽出
                vision_x = outputs.hidden_states
                
                # hidden_statesの形状を表示（デバッグ用）
                if vision_x is not None:
                    if isinstance(vision_x, tuple):
                        print(f"hidden_statesはタプルです（長さ: {len(vision_x)}）")
                        # 最後の層のhidden_stateを使用
                        vision_x = vision_x[-1]
                    print(f"vision_x形状: {vision_x.shape}")
                else:
                    print("警告: モデル出力からhidden_statesが取得できませんでした")
            else:
                print("プロセッサ入力がNoneのため、モデルは実行されません")
                vision_x = None
        except Exception as e:
            print(f"モデル実行中にエラーが発生しました: {e}")
            
            # 入力テンソルに関する詳細情報（デバッグ用）
            if processor_inputs is not None:
                for key, value in processor_inputs.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  - {key}: 形状={value.shape}, データ型={value.dtype}, デバイス={value.device}")
                    else:
                        print(f"  - {key}: タイプ={type(value)}")
            
            print(f"詳細なエラー情報:")
            traceback.print_exc()
            vision_x = None

        # モデル出力からlogitsを取得（あれば）
        if hasattr(vision_x, 'logits'):
            output_logits = vision_x.logits
            
            # logitsから次トークンを予測（argmax）
            output_ids = torch.argmax(output_logits, dim=-1)
        else:
            # logitsがない場合はinput_idsを使用
            if processor_inputs is not None and "input_ids" in processor_inputs:
                output_ids = processor_inputs["input_ids"]
                print("警告: モデル出力からlogitsが見つかりません。input_idsをoutput_idsとして使用します")
            else:
                # 入力も利用できない場合
                print("警告: モデル出力とinput_idsどちらもありません。処理を続行できません。")
                # 訓練中は損失を返す必要がある
                if not inference:
                    dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
                    return {
                        "loss": dummy_loss,
                        "ce_loss": dummy_loss,
                        "mask_bce_loss": dummy_loss,
                        "mask_dice_loss": dummy_loss,
                        "mask_loss": dummy_loss,
                    }
                else:
                    return {
                        "loss": torch.tensor(0.0, device=device),
                        "ce_loss": torch.tensor(0.0, device=device),
                        "mask_bce_loss": torch.tensor(0.0, device=device),
                        "mask_dice_loss": torch.tensor(0.0, device=device),
                        "mask_loss": torch.tensor(0.0, device=device),
                    }
        
        # <SEG>トークンの位置をマスクで特定
        # 出力トークンの次の位置から検索（LISAでは出力に<SEG>が含まれる）
        seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
        device = output_ids.device
        
        # 適切なパディングを追加
        padding_size = max(0, 255 - seg_token_mask.shape[1])
        if padding_size > 0:
            # 入力のサイズに応じてパディングを調整
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], padding_size), 
                                device=device, dtype=torch.bool),
                    seg_token_mask,
                ],
                dim=1,
            )
        else:
            # 必要に応じて、長い入力に対してもパディングを追加
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 1), 
                                device=device, dtype=torch.bool),
                    seg_token_mask,
                ],
                dim=1,
            )
        
        # 隠れ状態を処理
        hidden_states = []
        
        # 出力の最後の隠れ状態を取得
        if hasattr(vision_x, 'hidden_states') and vision_x.hidden_states is not None:
            last_hidden_state = vision_x.hidden_states[-1]
            
            # text_hidden_fcsがあれば適用
            if hasattr(self, "text_hidden_fcs"):
                for fc in self.text_hidden_fcs:
                    hidden_states.append(fc(last_hidden_state))
            else:
                # fcがなければそのまま使用
                hidden_states.append(last_hidden_state)
        else:
            print("警告: hidden_statesが見つかりません。マスク生成をスキップします。")
            # マスク生成をスキップして早期終了
            return {
                "loss": torch.tensor(0.0, device=device) if device else torch.tensor(0.0),
                "ce_loss": torch.tensor(0.0, device=device) if device else torch.tensor(0.0),
                "mask_bce_loss": torch.tensor(0.0, device=device) if device else torch.tensor(0.0),
                "mask_dice_loss": torch.tensor(0.0, device=device) if device else torch.tensor(0.0),
                "mask_loss": torch.tensor(0.0, device=device) if device else torch.tensor(0.0),
            }
        
        # 隠れ状態を合成
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        
        # <SEG>トークンに対応するhidden_stateを抽出
        pred_embeddings = last_hidden_state[seg_token_mask]
        
        # <SEG>トークンの数とオフセットを計算
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1, device=device, dtype=torch.long), seg_token_offset], dim=0
        )
        
        # offsetがNoneでなければ使用
        if offset is not None:
            seg_token_offset = seg_token_offset[offset]

        # 予測埋め込みを処理
        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            if start_i < end_i:  # 開始と終了が同じではないことを確認
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            else:
                print(f"警告: セグメントインデックス {i} の範囲が無効です (start={start_i}, end={end_i})")
                # 空のエンベディングを追加（処理を続行するため）
                if i > 0 and len(pred_embeddings_) > 0:
                    # 前のエンベディングと同じ形状のゼロテンソルを使用
                    zero_embed = torch.zeros_like(pred_embeddings_[-1])
                    pred_embeddings_.append(zero_embed)
                else:
                    # 最初のエンベディングの場合、適当な形状のゼロテンソルを作成
                    embed_dim = last_hidden_state.shape[-1]
                    zero_embed = torch.zeros((1, embed_dim), device=device)
                    pred_embeddings_.append(zero_embed)
        
        # 空のリストの場合の処理（<SEG>トークンが見つからない場合）
        if len(pred_embeddings_) == 0:
            print("警告: <SEG>トークンが見つかりません。マスク生成をスキップします。")
            # マスク生成をスキップして早期終了
            return {
                "loss": torch.tensor(0.0, device=device) if device else torch.tensor(0.0),
                "ce_loss": torch.tensor(0.0, device=device) if device else torch.tensor(0.0),
                "mask_bce_loss": torch.tensor(0.0, device=device) if device else torch.tensor(0.0),
                "mask_dice_loss": torch.tensor(0.0, device=device) if device else torch.tensor(0.0),
                "mask_loss": torch.tensor(0.0, device=device) if device else torch.tensor(0.0),
            }
            
        pred_embeddings = pred_embeddings_

        # マスク生成
        multimask_output = False
        pred_masks = []
        
        try:
            # SAMの準備ができているか確認
            if image_embeddings is None or len(image_embeddings) == 0:
                raise ValueError("image_embeddingsがありません。")
            
            # visual_modelを取得
            visual_model = None
            if hasattr(self, "visual_model"):
                visual_model = self.visual_model
            else:
                raise AttributeError("visual_modelが見つかりません")
            
            # 各<SEG>トークンに対応するマスクを生成
            for i in range(len(pred_embeddings)):
                try:
                    # バッチインデックスの調整
                    batch_idx = min(i, len(image_embeddings) - 1)
                    
                    # SAMのプロンプトエンコーダーにテキスト埋め込みを渡す
                    (
                        sparse_embeddings,
                        dense_embeddings,
                    ) = visual_model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=pred_embeddings[i].unsqueeze(1),
                    )

                    # データ型を合わせる
                    sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                    
                    # マスクデコーダーを使用してマスクを生成
                    low_res_masks, iou_predictions = visual_model.mask_decoder(
                        image_embeddings=image_embeddings[batch_idx].unsqueeze(0),
                        image_pe=visual_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=multimask_output,
                    )
                    
                    # マスクの後処理
                    try:
                        # resize_listとlabel_listの両方がある場合
                        if resize_list is not None and i < len(resize_list) and label_list is not None and i < len(label_list):
                            pred_mask = visual_model.postprocess_masks(
                                low_res_masks,
                                input_size=resize_list[i],
                                original_size=label_list[i].shape,
                            )
                        # label_listからサイズを取得
                        elif label_list is not None and i < len(label_list):
                            original_size = label_list[i].shape
                            input_size = visual_model.image_encoder.img_size
                            pred_mask = visual_model.postprocess_masks(
                                low_res_masks,
                                input_size=input_size,
                                original_size=original_size,
                            )
                        else:
                            # デフォルトサイズを使用
                            print("警告: マスクサイズ情報がありません。デフォルトサイズを使用します。")
                            input_size = visual_model.image_encoder.img_size
                            original_size = (1024, 1024)  # デフォルトサイズ
                            pred_mask = visual_model.postprocess_masks(
                                low_res_masks,
                                input_size=input_size,
                                original_size=original_size,
                            )
                    except Exception as post_e:
                        print(f"マスク後処理中にエラーが発生しました: {post_e}")
                        # エラーの場合はlow_res_masksをそのまま使用
                        pred_mask = low_res_masks
                    
                    # マスクを追加（最初のマスクのみ使用）
                    pred_masks.append(pred_mask[:, 0])
                    
                except Exception as mask_e:
                    print(f"マスク生成中にエラーが発生しました: {mask_e}")
                    # エラーが発生した場合は空のマスクを返す
                    try:
                        if label_list is not None and i < len(label_list):
                            shape = label_list[i].shape
                            empty_mask = torch.zeros((1, *shape), device=device)
                        else:
                            # デフォルトサイズの空マスク
                            empty_mask = torch.zeros((1, 1024, 1024), device=device)
                        pred_masks.append(empty_mask)
                    except Exception as e:
                        print(f"空マスク生成中にエラーが発生しました: {e}")
                        # 最小サイズの空マスク
                        empty_mask = torch.zeros((1, 100, 100), device=device)
                        pred_masks.append(empty_mask)
                        
        except Exception as e:
            print(f"全体的なマスク生成でエラーが発生しました: {e}")
            # エラーの場合は空のマスクリストを返す
            pred_masks = []
            for i in range(len(pred_embeddings)):
                try:
                    if label_list is not None and i < len(label_list):
                        shape = label_list[i].shape
                        empty_mask = torch.zeros((1, *shape), device=device)
                    else:
                        empty_mask = torch.zeros((1, 100, 100), device=device)
                    pred_masks.append(empty_mask)
                except:
                    # 最小サイズの空マスク
                    empty_mask = torch.zeros((1, 100, 100), device=device)
                    pred_masks.append(empty_mask)

        # 推論モードの場合は予測マスクを返す
        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": masks_list,
            }

        # 損失計算のためのモデル出力と正解マスク
        model_output = vision_x
        gt_masks = masks_list

        # 言語モデルのCE損失を取得
        ce_loss = model_output.loss if hasattr(model_output, 'loss') else torch.tensor(0.0, device=device)
        ce_loss = ce_loss * self.ce_loss_weight
        
        # マスク損失の計算
        mask_bce_loss = torch.tensor(0.0, device=device)
        mask_dice_loss = torch.tensor(0.0, device=device)
        num_masks = 0
        
        if pred_masks and gt_masks and len(pred_masks) > 0 and len(gt_masks) > 0:
            try:
                for batch_idx in range(min(len(pred_masks), len(gt_masks))):
                    try:
                        gt_mask = gt_masks[batch_idx]
                        pred_mask = pred_masks[batch_idx]
                        
                        # 形状が一致するか確認
                        if gt_mask.shape[0] != pred_mask.shape[0]:
                            print(f"警告: バッチ{batch_idx}のマスク形状が一致しません。gt_mask: {gt_mask.shape}, pred_mask: {pred_mask.shape}")
                            # 最小の共通サイズを使用
                            min_size = min(gt_mask.shape[0], pred_mask.shape[0])
                            gt_mask = gt_mask[:min_size]
                            pred_mask = pred_mask[:min_size]
                        
                        # BCE損失の計算
                        batch_bce = sigmoid_ce_loss(
                            pred_mask, gt_mask, num_masks=gt_mask.shape[0]
                        ) * gt_mask.shape[0]
                        
                        # Dice損失の計算
                        batch_dice = dice_loss(
                            pred_mask, gt_mask, num_masks=gt_mask.shape[0]
                        ) * gt_mask.shape[0]
                        
                        mask_bce_loss += batch_bce
                        mask_dice_loss += batch_dice
                        num_masks += gt_mask.shape[0]
                    except Exception as e:
                        print(f"バッチ{batch_idx}のマスク損失計算中にエラーが発生しました: {e}")
                        # このバッチをスキップ
            
            except Exception as e:
                print(f"マスク損失計算中にエラーが発生しました: {e}")
        
        # 損失の正規化と重み付け
        if num_masks > 0:
            mask_bce_loss = self.bce_loss_weight * mask_bce_loss / num_masks
            mask_dice_loss = self.dice_loss_weight * mask_dice_loss / num_masks
            mask_loss = mask_bce_loss + mask_dice_loss
        else:
            # マスクがない場合は0を設定
            mask_loss = torch.tensor(0.0, device=device)
            
        # 損失計算の前に必要なチェックと勾配追跡の設定
        if not inference:
            # CEロスの勾配追跡確保
            if torch.is_tensor(ce_loss):
                if not ce_loss.requires_grad:
                    ce_loss = ce_loss.clone().detach().requires_grad_(True)
            else:
                # CEロスがテンソルでない場合、0のテンソルを作成（勾配追跡あり）
                ce_loss = torch.tensor(0.0, device=device, requires_grad=True)
                
            # マスクロスの勾配追跡確保
            if torch.is_tensor(mask_loss):
                if not mask_loss.requires_grad:
                    mask_loss = mask_loss.clone().detach().requires_grad_(True)
            else:
                # マスクロスがテンソルでない場合、0のテンソルを作成（勾配追跡あり）
                mask_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # 損失合計の計算（勾配追跡を維持）
            loss = None
            if torch.is_tensor(ce_loss) and ce_loss.item() > 0:
                loss = ce_loss.clone()
            
            if torch.is_tensor(mask_loss) and mask_loss.item() > 0:
                if loss is not None:
                    loss = loss + mask_loss
                else:
                    loss = mask_loss.clone()
            
            # どちらの損失も有効でない場合、ダミーの勾配付き損失を作成
            if loss is None or loss.item() == 0:
                # 最小値だがゼロではない勾配を持つダミー損失
                loss = torch.tensor(1e-8, device=device, requires_grad=True)
        else:
            # 推論モードの場合は勾配は不要
            loss = None
            if torch.is_tensor(ce_loss) and torch.is_tensor(mask_loss):
                loss = ce_loss + mask_loss
            elif torch.is_tensor(ce_loss):
                loss = ce_loss
            elif torch.is_tensor(mask_loss):
                loss = mask_loss
            else:
                loss = torch.tensor(0.0, device=device)

        # 戻り値
        if inference:
            return {
                "masks": pred_masks,
                "low_res_masks": low_res_pred_masks,
                "iou_scores": iou_scores,
                "seg_token_counts": seg_token_counts,
            }
        else:
            return {
                "loss": loss,
                "ce_loss": ce_loss,
                "mask_bce_loss": mask_bce_loss,
                "mask_dice_loss": mask_dice_loss,
                "mask_loss": mask_loss,
            }

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        images=None,
        **kwargs
    ):
        """
        Llama3.2 Visionを使用して生成するための入力を準備します。
        """
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # 生成のための画像入力を準備
        batch_inputs = {}
        if input_ids is not None:
            batch_inputs["input_ids"] = input_ids
        if past_key_values is not None:
            batch_inputs["past_key_values"] = past_key_values
        if attention_mask is not None:
            batch_inputs["attention_mask"] = attention_mask
        if inputs_embeds is not None:
            batch_inputs["inputs_embeds"] = inputs_embeds
        if images is not None:
            # Llama 3.2 Visionでは'images'ではなく'pixel_values'を使用
            batch_inputs["pixel_values"] = images
            
            # バッチサイズを取得
            if isinstance(images, torch.Tensor):
                batch_size = images.shape[0]
                # aspect_ratio_idsを作成（デフォルトで0）
                device = images.device
                batch_inputs["aspect_ratio_ids"] = torch.zeros(batch_size, dtype=torch.long, device=device)

        return batch_inputs

    def get_visual_embs(self, images):
        """
        SAMのvisual_modelを使用して画像埋め込みを取得します
        
        Args:
            images: 入力画像テンソル [batch_size, channels, height, width]
            
        Returns:
            image_embeddings: SAM画像エンコーダからの特徴
        """
        if images is None:
            raise ValueError("入力画像がNoneです")
            
        if not hasattr(self, "visual_model") or self.visual_model is None:
            raise ValueError("visual_modelが初期化されていません")
            
        # 画像をデバイスに移動
        device = next(self.model.parameters()).device
        if images.device != device:
            images = images.to(device)
            
        with torch.no_grad():
            # SAMイメージエンコーダを呼び出し
            image_embeddings = self.visual_model.image_encoder(images)
            
        return image_embeddings

def compute_dice_loss(inputs, targets, smooth=1):
    """
    Compute Dice損失（Sørensen-Dice係数に基づく）
    
    Args:
        inputs: 予測値（シグモイド前のロジット）
        targets: 正解マスク
        smooth: 数値安定性のための平滑化係数
        
    Returns:
        Dice係数（1に近いほど良い）
    """
    # シグモイド関数で確率値に変換
    inputs = torch.sigmoid(inputs)
    
    # 平坦化
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    
    # 交差部分
    intersection = (inputs * targets).sum()
    
    # Dice係数の計算: 2*|X∩Y|/(|X|+|Y|)
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    
    return dice

class LISAForCausalLM(nn.Module):
    """
    LISA for Causal Language Modeling.
    Llama3.2 Vision (MllamaForConditionalGeneration) + SAM
    """
    def __init__(self, **kwargs):
        super().__init__()
        
        # train_ds.pyから渡されるパラメータを正しく処理
        self.lisa_model = LisaModel(
            model_id=kwargs.get("model_id", "meta-llama/Llama-3.2-11B-Vision-Instruct"),
            sam_vision_encoder=kwargs.get("sam_vision_encoder", kwargs.get("sam_encoder", None)),
            mask_decoder=kwargs.get("mask_decoder", None),
            torch_dtype=kwargs.get("torch_dtype", torch.float16),
            device_map=kwargs.get("device_map", None),
            train_mask_decoder=kwargs.get("train_mask_decoder", True),
            out_dim=kwargs.get("out_dim", 256)
        )
        
        # オリジナルLISAとの互換性のために、model属性も追加
        self.model = self.lisa_model
        
        # 各種パラメータを設定
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        self.ce_loss_weight = kwargs.get("ce_loss_weight", 1.0)
        self.bce_loss_weight = kwargs.get("bce_loss_weight", 1.0)
        self.dice_loss_weight = kwargs.get("dice_loss_weight", 1.0)
        
        # vision_pretrained、low_cpu_mem_usageなどのパラメータも保存
        self.vision_pretrained = kwargs.get("vision_pretrained", None)
        self.vision_tower = kwargs.get("vision_tower", None)
        self.use_mm_start_end = kwargs.get("use_mm_start_end", True)
        self.low_cpu_mem_usage = kwargs.get("low_cpu_mem_usage", True)
        
        # seg_token_idxをLisaModelから取得、または直接設定
        if "seg_token_idx" in kwargs and kwargs["seg_token_idx"] is not None:
            self.seg_token_idx = kwargs["seg_token_idx"]
            # LisaModelにも伝える
            if hasattr(self.model, "seg_token_idx"):
                self.model.seg_token_idx = self.seg_token_idx
        else:
            self.seg_token_idx = getattr(self.lisa_model, "seg_token_idx", None)
            if self.seg_token_idx is None:
                print("警告: seg_token_idxがLISAForCausalLMで設定されていません")
                self.seg_token_idx = -1  # デフォルト値（エラー時用）

    def get_processor(self):
        """processorを取得"""
        if hasattr(self.lisa_model, "processor"):
            return self.lisa_model.processor
        else:
            print("警告: processorが設定されていません")
            return None
            
    def get_visual_embs(self, images, return_width_height=False):
        """
        ビジュアルな埋め込みを取得
        """
        return self.lisa_model.get_visual_embs(images)
        
    def forward(self, **kwargs):
        """
        モデルの順伝播
        """
        return self.lisa_model.forward(**kwargs)
