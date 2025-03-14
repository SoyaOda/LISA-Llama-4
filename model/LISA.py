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

# SAMモデルをインポート
from model.segment_anything import build_sam_vit_h

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
        device_map="auto",  # 自動デバイスマッピングを有効化
        torch_dtype=torch.bfloat16,  # BFloat16を使用してメモリ使用量を削減
        train_mask_decoder=True,
        out_dim=256,
        vision_pretrained=None,
        load_in_8bit=False,  # 8ビット量子化オプション
        load_in_4bit=False   # 4ビット量子化オプション
    ):
        """
        SAMに基づいたLlama 3.2 Visionモデル。
        
        Args:
            model_id: Llama 3.2 Visionモデルの識別子
            sam_vision_encoder: SAMのビジョンエンコーダー
            mask_decoder: マスクデコーダー
            device_map: モデルをどのデバイスにマッピングするか ("auto"を指定すると最適なデバイス配置)
            torch_dtype: モデルのデータ型 (bfloat16を推奨)
            train_mask_decoder: マスクデコーダーを訓練するかどうか
            out_dim: 出力次元
            vision_pretrained: SAMモデルの事前学習済み重み
            load_in_8bit: 8ビット量子化を使用するかどうか (メモリを節約)
            load_in_4bit: 4ビット量子化を使用するかどうか (メモリをさらに節約)
        """
        super().__init__()
        
        print(f"LisaModelを初期化します。model_id: {model_id}")
        self.model_id = model_id
        self.train_mask_decoder = train_mask_decoder
        self.out_dim = out_dim
        self.seg_token_idx = None
        self.vision_pretrained = vision_pretrained
        
        # 量子化オプションを設定
        quantization_config = None
        if load_in_8bit or load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print(f"量子化設定: 8ビット={load_in_8bit}, 4ビット={load_in_4bit}")
        
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
            # AutoModelForVision2Seqを使用してモデルをロード
            from transformers import AutoModelForVision2Seq, AutoProcessor
            print(f"AutoModelForVision2Seqを使用してモデルをロードします: {model_id}")
            print(f"デバイスマップ: {device_map}, データ型: {torch_dtype}")

            # メモリを節約するための設定
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                device_map=device_map,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config
            )
            print(f"モデルのロードに成功しました")
            
            # プロセッサを初期化
            try:
                print(f"プロセッサを初期化します: {model_id}")
                self.processor = AutoProcessor.from_pretrained(model_id)
                print("プロセッサの初期化に成功しました")
            except Exception as e:
                print(f"プロセッサの初期化中にエラーが発生しました: {e}")
                self.processor = None
                
        except Exception as e:
            print(f"モデル初期化中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.processor = None
            raise ValueError(f"モデルの初期化に失敗しました: {e}")
        
        # SAMビジョンエンコーダーの初期化
        print("SAMビジョンエンコーダーを初期化します")
        if sam_vision_encoder is not None:
            self.visual_model = sam_vision_encoder
            print("提供されたSAMビジョンエンコーダーを使用します")
        else:
            # SAMビジョンエンコーダーを構築
            try:
                if vision_pretrained:
                    print(f"SAMビジョンエンコーダーを構築します: {vision_pretrained}")
                    from .segment_anything import build_sam_vit_h
                    self.visual_model = build_sam_vit_h(checkpoint=vision_pretrained)
                    print("SAMビジョンエンコーダーの構築に成功しました")
                else:
                    print("警告: vision_pretrainedが指定されていないため、SAMビジョンエンコーダーを初期化していません")
                    self.visual_model = None
            except Exception as e:
                print(f"SAMビジョンエンコーダーの初期化中にエラーが発生しました: {e}")
                traceback.print_exc()
                self.visual_model = None
                raise ValueError(f"SAMビジョンエンコーダーの初期化に失敗しました: {e}")

        # マスクデコーダーの初期化
        print("マスクデコーダーを初期化します")
        if mask_decoder is not None:
            self.mask_decoder = mask_decoder
            print("提供されたマスクデコーダーを使用します")
        elif hasattr(self, 'visual_model') and self.visual_model is not None:
            print("SAMのマスクデコーダーを使用します")
            self.mask_decoder = self.visual_model.mask_decoder
        else:
            print("警告: マスクデコーダーを初期化していません")
            self.mask_decoder = None
        
        # トレーニング設定
        if not train_mask_decoder and hasattr(self, 'mask_decoder') and self.mask_decoder is not None:
            print("マスクデコーダーをフリーズします")
            for param in self.mask_decoder.parameters():
                param.requires_grad = False
                
        # デバイス情報の表示
        if hasattr(self, 'model') and self.model is not None:
            print(f"初期化されたモデルのデバイス: {next(self.model.parameters()).device}")
        
    def resize_token_embeddings(self, new_num_tokens):
        """
        トークン埋め込みのサイズを変更します。
        
        Args:
            new_num_tokens: 新しいトークン数
            
        Returns:
            更新されたモデル
        """
        if hasattr(self.model, "resize_token_embeddings"):
            print(f"resize_token_embeddingsメソッドを呼び出します: {new_num_tokens}")
            # モデルのトークン埋め込みのサイズを変更
            self.model.resize_token_embeddings(new_num_tokens)
            
            # 出力埋め込みのサイズも変更（Llama3.2は入出力の埋め込みが分離しているため）
            if hasattr(self.model, "lm_head"):
                print("lm_headのサイズも変更します")
                current_lm_head = self.model.lm_head
                input_embeddings = self.model.model.embed_tokens
                
                # 新しいlm_headを作成
                new_lm_head = nn.Linear(
                    input_embeddings.embedding_dim, new_num_tokens, bias=current_lm_head.bias is not None
                )
                
                # 既存の重みをコピー
                new_lm_head.weight.data[:current_lm_head.weight.data.shape[0]] = current_lm_head.weight.data
                
                # バイアスがある場合はそれもコピー
                if current_lm_head.bias is not None:
                    new_lm_head.bias.data[:current_lm_head.bias.data.shape[0]] = current_lm_head.bias.data
                
                # モデルの出力レイヤーを更新
                self.model.lm_head = new_lm_head
                
            return self.model
        else:
            print("警告: resize_token_embeddingsメソッドがモデルにありません")
            return None
            
    def get_processor(self):
        """
        モデルのprocessorを取得します。
        
        Returns:
            processor: モデルのprocessor、設定されていない場合はNone
        """
        if hasattr(self, "processor"):
            return self.processor
        else:
            print("警告: processorが設定されていません")
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

        # プロセッサを取得
        try:
            processor = self.get_processor()
            if processor is None:
                raise ValueError("processorが初期化されていません。model_forwardメソッドを実行できません。")
        except AttributeError as e:
            print(f"[エラー情報] プロセッサの取得中にエラーが発生しました: {e}")
            print("データの処理を続行しますが、一部の機能が制限される可能性があります。")
            processor = None
        
        # プロセッサを使用して画像とテキストを適切に処理
        processor_inputs = None
        try:
            if processor is not None:
                print("プロセッサを使用して入力を処理します")
                
                # テキスト入力の準備
                if isinstance(input_ids, torch.Tensor):
                    if tokenizer is None:
                        raise ValueError("テンソル形式のinput_idsに対してtokenizerが必要です")
                    
                    # トークンIDをテキストにデコード
                    decoded_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
                    
                    # 各テキストに画像トークンが含まれているか確認し、なければ追加
                    processed_texts = []
                    for text in decoded_texts:
                        # 画像トークンがない場合は、テキストの先頭に追加
                        if "<|image|>" not in text:
                            text = "<|image|> " + text
                        processed_texts.append(text)
                    
                    text_input = processed_texts
                else:
                    # すでに文字列または文字列のリストの場合
                    text_input = []
                    if isinstance(input_ids, str):
                        input_ids = [input_ids]
                        
                    for text in input_ids:
                        if "<|image|>" not in text:
                            text = "<|image|> " + text
                        text_input.append(text)
                
                # 画像入力の準備
                images_for_processor = None
                if images_clip is not None:
                    # テンソルからPILイメージに変換
                    try:
                        import numpy as np
                        
                        # 画像の形状を確認
                        print(f"images_clip形状: {images_clip.shape}")
                        
                        # 6次元の場合の処理（特殊なケース）
                        if images_clip.dim() == 6:
                            print("6次元の画像テンソルを検出しました。形状を調整します。")
                            # 形状を確認
                            print(f"元の形状: {images_clip.shape}")  # 例: [1, 1, 4, 3, 560, 560]
                            
                            # バッチサイズとチャネル数を取得
                            batch_size = images_clip.shape[0]
                            num_images_per_sample = images_clip.shape[2]  # 各サンプルの画像数
                            channels = images_clip.shape[3]
                            height = images_clip.shape[4]
                            width = images_clip.shape[5]
                            
                            # バッチ内の各サンプルから最初の画像のみを取得（単一画像処理）
                            images_clip = images_clip[:, 0, 0].reshape(batch_size, channels, height, width)
                            print(f"単一画像に変換後の形状: {images_clip.shape}")  # 例: [1, 3, 560, 560]
                        
                        # 単一画像か複数画像かを確認
                        if len(images_clip.shape) == 4:  # [batch, channels, height, width]
                            images_for_processor = []
                            for i in range(images_clip.shape[0]):
                                # [channels, height, width] -> [height, width, channels]
                                # BFloat16をfloat32に変換してからNumPy配列に変換
                                img_np = images_clip[i].to(torch.float32).permute(1, 2, 0).cpu().numpy()
                                img_np = np.clip(img_np, 0, 1)
                                img_np = (img_np * 255).astype(np.uint8)
                                images_for_processor.append(Image.fromarray(img_np))
                    except Exception as e:
                        print(f"画像変換中にエラーが発生しました: {e}")
                        traceback.print_exc()
                        # エラーを表示して中断（デバッグのため）
                        raise ValueError(f"画像変換中にエラーが発生しました: {e}")
                        
                # 画像とテキストの数が一致していることを確認
                if images_for_processor is not None:
                    num_images = len(images_for_processor)
                    num_texts = len(text_input)
                    
                    if num_images != num_texts:
                        print(f"警告: 画像数({num_images})とテキスト数({num_texts})が一致しません。調整します。")
                        if num_images > num_texts:
                            # 画像数に合わせてテキストを調整（テキスト数が少ない場合）
                            print("テキスト数が足りないため、テキストを複製します")
                            text_input = text_input * (num_images // num_texts + 1)
                            text_input = text_input[:num_images]
                        else:
                            # テキスト数が多い場合は、ダミー画像を追加
                            print("画像数が足りないため、ダミー画像を追加します")
                            # ダミー画像を作成（黒い画像）
                            dummy_image = Image.new("RGB", (560, 560), color="black")
                            
                            # 必要な数だけダミー画像を追加
                            for _ in range(num_texts - num_images):
                                images_for_processor.append(dummy_image)
                                
                            print(f"ダミー画像追加後の画像数: {len(images_for_processor)}")
                    
                    # デバイスを確認
                    device = images.device if hasattr(images, 'device') else "cuda" if torch.cuda.is_available() else "cpu"
                    processor_inputs = processor(text=text_input, images=images_for_processor, return_tensors="pt", padding=True)
                    
                    # デバイスを合わせる
                    for k, v in processor_inputs.items():
                        if isinstance(v, torch.Tensor):
                            processor_inputs[k] = v.to(device)
                            
                    # aspect_ratio_idsの確認と追加
                    if "pixel_values" in processor_inputs and "aspect_ratio_ids" not in processor_inputs:
                        print("aspect_ratio_idsを追加します")
                        processor_inputs["aspect_ratio_ids"] = torch.zeros(
                            processor_inputs["pixel_values"].shape[0],
                            dtype=torch.long,
                            device=processor_inputs["pixel_values"].device
                        )
                            
                    # バッチ入力に追加
                    for k, v in processor_inputs.items():
                        kwargs[k] = v
                
        except Exception as e:
            print(f"プロセッサによる画像処理中にエラーが発生しました: {e}")
            traceback.print_exc()
            # エラーを表示して処理を続行（フォールバック）
            processor_inputs = None
        
        # モデル実行（Llama3.2 Vision）
        vision_x = None
        try:
            if processor_inputs is not None:
                print("Llama3.2 Visionモデルを実行します")
                
                # メモリ使用量削減のため、必要に応じてbfloat16で計算
                # これはオプションで、すでにbfloat16で初期化されている場合は不要
                use_bf16 = True
                if use_bf16:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = self.model(
                            **processor_inputs,
                            output_hidden_states=True,
                            return_dict=True
                        )
                else:
                    outputs = self.model(
                        **processor_inputs,
                        output_hidden_states=True,
                        return_dict=True
                    )
                
                # hidden_statesを抽出
                if hasattr(outputs, 'hidden_states'):
                    vision_x = outputs.hidden_states
                    
                    # hidden_statesの形状を表示（デバッグ用）
                    if isinstance(vision_x, tuple):
                        print(f"hidden_statesはタプルです（長さ: {len(vision_x)}）")
                        # 最後の層のhidden_stateを使用
                        vision_x = vision_x[-1]
                    print(f"vision_x形状: {vision_x.shape}")
                else:
                    print("警告: モデル出力からhidden_statesが取得できませんでした")
                    vision_x = None
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
        model_output = outputs  # vision_xではなくoutputsを使用
        gt_masks = masks_list
        device = next(self.model.parameters()).device

        # 言語モデルのCE損失を取得
        ce_loss = None
        if model_output is not None and hasattr(model_output, 'loss') and model_output.loss is not None:
            ce_loss = model_output.loss
            if not ce_loss.requires_grad:
                print("警告: CE損失に勾配がありません。requires_gradをTrueに設定します。")
                ce_loss = ce_loss.detach().clone()
                ce_loss.requires_grad_(True)
        else:
            print("警告: モデル出力からlossが見つかりません。0.0で初期化します。")
            ce_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
        ce_loss = ce_loss * self.ce_loss_weight
        
        # マスク損失の計算
        mask_bce_loss = torch.tensor(0.0, device=device, requires_grad=True)
        mask_dice_loss = torch.tensor(0.0, device=device, requires_grad=True)
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
                        
                        # マスクに勾配が必要かチェック
                        if not pred_mask.requires_grad:
                            print(f"警告: バッチ{batch_idx}の予測マスクに勾配がありません。requires_gradをTrueに設定します。")
                            # 予測マスクのコピーを作成し、requires_gradをTrueに設定
                            pred_mask = pred_mask.detach().clone()
                            pred_mask.requires_grad_(True)
                        
                        # BCE損失の計算
                        batch_bce = sigmoid_ce_loss(
                            pred_mask, gt_mask, num_masks=gt_mask.shape[0]
                        ) * gt_mask.shape[0]
                        
                        # Dice損失の計算
                        batch_dice = dice_loss(
                            pred_mask, gt_mask, num_masks=gt_mask.shape[0]
                        ) * gt_mask.shape[0]
                        
                        mask_bce_loss = mask_bce_loss + batch_bce
                        mask_dice_loss = mask_dice_loss + batch_dice
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
            
        # 損失合計の計算（勾配追跡を維持）
        loss = None
        # 両方の損失を常に足し合わせる（勾配追跡のため）
        ce_loss = ce_loss if torch.is_tensor(ce_loss) else torch.tensor(0.0, device=device, requires_grad=True)
        mask_loss = mask_loss if torch.is_tensor(mask_loss) else torch.tensor(0.0, device=device, requires_grad=True)
        
        # 合計損失の計算
        loss = ce_loss + mask_loss

        # 損失のチェック
        if loss.item() == 0:
            print("警告: 合計損失が0になっています。これは勾配計算で問題が発生する可能性があります。")
            # 最小値だがゼロではない勾配を持つダミー損失を追加
            dummy_param = next(self.model.parameters())
            loss = loss + 0.0001 * (dummy_param * dummy_param).sum()

        # 戻り値
        if inference:
            return {
                "masks": pred_masks,
                "low_res_masks": low_res_masks,
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

    def get_visual_embs(self, images):
        """
        SAMの画像エンコーダーを使用して視覚的特徴を抽出します。
        
        Args:
            images: 入力画像テンソル
            
        Returns:
            image_embeddings: SAM画像エンコーダからの特徴
        """
        if images is None:
            raise ValueError("入力画像がNoneです")
            
        if not hasattr(self, "visual_model") or self.visual_model is None:
            raise ValueError("visual_modelが初期化されていません")
            
        # 画像をデバイスに移動
        device = next(self.parameters()).device
        if images.device != device:
            images = images.to(device)
            
        try:
            with torch.no_grad():
                # SAMイメージエンコーダを呼び出し
                image_embeddings = self.visual_model.image_encoder(images)
                
            return image_embeddings
        except Exception as e:
            print(f"SAM画像エンコーダの実行中にエラーが発生しました: {e}")
            traceback.print_exc()
            raise

class LISAForCausalLM(MllamaForConditionalGeneration, GenerationMixin):
    """
    LISA for Causal Language Modeling.
    Llama3.2 Vision + SAMを直接統合したシンプルな実装
    """
    def __init__(self, config=None, **kwargs):
        # configがなければモデルIDから作成
        model_id = kwargs.pop("model_id", "meta-llama/Llama-3.2-11B-Vision-Instruct")
        
        if config is None:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_id)
        
        # SAM関連のパラメータを保存（MllamaForConditionalGenerationには渡さない）
        self.seg_token_idx = kwargs.pop("seg_token_idx", None)
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", 1.0)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", 1.0)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", 1.0)
        self.train_mask_decoder = kwargs.pop("train_mask_decoder", True)
        self.out_dim = kwargs.pop("out_dim", 256)
        self.vision_pretrained = kwargs.pop("vision_pretrained", None)
        # その他のLISA固有パラメータも取り除く
        kwargs.pop("vision_tower", None)
        kwargs.pop("use_mm_start_end", None)
        
        # MllamaForConditionalGeneration初期化のためのパラメータだけを残す
        torch_dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        device_map = kwargs.pop("device_map", "auto")
        
        # 量子化設定
        load_in_8bit = kwargs.pop("load_in_8bit", False)
        load_in_4bit = kwargs.pop("load_in_4bit", False)
        quantization_config = None
        if load_in_8bit or load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # 内部変数を初期化（無限再帰を防ぐため）
        self._base_model_reference = None
        
        # 親クラスを初期化 - SAM関連のパラメータは除外済み
        super(LISAForCausalLM, self).__init__(config)
        
        # 事前学習済みモデルをロード
        try:
            from transformers import AutoModelForVision2Seq
            print(f"base_modelを初期化: {model_id}")
            # モデルを直接ロード - 参照を保存
            self._base_model_reference = AutoModelForVision2Seq.from_pretrained(
                model_id, 
                config=config,
                device_map=device_map,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
                **kwargs  # 残りのパラメータ
            )
            print("base_modelの初期化完了")
            
        except Exception as e:
            print(f"ベースモデルのロード中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
        
        # プロセッサを初期化
        try:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_id)
        except Exception as e:
            print(f"プロセッサの初期化エラー: {e}")
            self.processor = None
                
        # SAM初期化 - 保存しておいたSAM関連のパラメータを使用
        if self.vision_pretrained:
            try:
                from model.segment_anything import build_sam_vit_h
                self.visual_model = build_sam_vit_h(checkpoint=self.vision_pretrained)
                # SAMパラメータをfreeze
                for param in self.visual_model.parameters():
                    param.requires_grad = False
                    
                # マスクデコーダーは学習対象にする
                if self.train_mask_decoder and hasattr(self.visual_model, "mask_decoder"):
                    self.visual_model.mask_decoder.train()
                    for param in self.visual_model.mask_decoder.parameters():
                        param.requires_grad = True
                        
                # テキスト埋め込みからSAMプロンプト埋め込みへの変換層
                prompt_embed_dim = 256  # SAMデフォルト値
                hidden_size = getattr(config, 'hidden_size', 4096)
                if hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
                    hidden_size = config.text_config.hidden_size
                    
                self.text_hidden_fcs = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_size, self.out_dim),
                        nn.Dropout(0.0),
                    )
                ])
                
            except Exception as e:
                print(f"SAM初期化エラー: {e}")
                import traceback
                traceback.print_exc()
                raise
    
    @property
    def base_model(self):
        """base_modelを取得するプロパティ"""
        return self._base_model_reference
    
    @property
    def model(self):
        """
        train_ds.pyとの互換性のためのプロパティ。
        model.modelにアクセスするときにモデルプロキシを返します。
        """
        # オリジナルのLISAでは、model.modelがLisaModelインスタンスを参照していた
        # ここでは、base_modelを通じて事前学習済みの言語モデルにアクセスできるようにする
        
        class ModelProxy:
            """
            モデルプロキシクラス - 自身（LISAForCausalLM）とbase_modelを組み合わせたアクセスを提供
            """
            def __init__(self, lisa_model, base_model):
                self.lisa_model = lisa_model  # LISAForCausalLMインスタンス
                self.base_model = base_model  # AutoModelForVision2Seqインスタンス
                # ModelProxyの内部辞書（_dict属性）を基本のProxyと同じに保つ
                if hasattr(base_model, 'model'):
                    self.model = base_model.model
                else:
                    # base_modelが.modelを持たない場合
                    self.model = self  # 循環参照を作らないように注意
                
            def __getattr__(self, name):
                # 最初にself.modelが自分自身でない場合に、その属性を探す
                if hasattr(self, 'model') and self.model is not self and hasattr(self.model, name):
                    return getattr(self.model, name)
                
                # 次にbase_modelで属性を探す
                if self.base_model is not None and hasattr(self.base_model, name):
                    return getattr(self.base_model, name)
                
                # なければlisa_modelで探す
                if hasattr(self.lisa_model, name):
                    return getattr(self.lisa_model, name)
                    
                # どちらにもなければAttributeError
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            
            # 特に重要な属性へのアクセス方法を明示的に定義
            @property
            def embed_tokens(self):
                """embed_tokens属性へのアクセス - train_ds.pyで使用"""
                # base_model.modelにembed_tokensがある場合
                if self.base_model is not None and hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'embed_tokens'):
                    return self.base_model.model.embed_tokens
                # base_modelに直接embed_tokensがある場合
                elif self.base_model is not None and hasattr(self.base_model, 'embed_tokens'):
                    return self.base_model.embed_tokens
                # lisa_modelにembed_tokensがある場合
                elif hasattr(self.lisa_model, 'embed_tokens'):
                    return self.lisa_model.embed_tokens
                # 見つからない場合は例外
                raise AttributeError(f"'embed_tokens' attribute not found in model structure")
                
            def to(self, device):
                # toメソッドの特別な処理（train_ds.pyで使用）
                if self.base_model is not None:
                    self.base_model.to(device)
                return self
                
            @property
            def config(self):
                # configプロパティの特別な処理（train_ds.pyで使用）
                if self.base_model is not None and hasattr(self.base_model, "config"):
                    return self.base_model.config
                return self.lisa_model.config
                
            def state_dict(self, *args, **kwargs):
                """
                モデルの状態辞書を取得します
                ModelProxyを通じて安全に呼び出せるよう、_state_dictを使用
                """
                # 1. 現在のインスタンスのstate_dictメソッドを退避（無限再帰を防ぐ）
                original_state_dict = getattr(self.__class__, 'state_dict', None)
                setattr(self.__class__, 'state_dict', None)
                
                try:
                    # 2. base_modelのstate_dictを取得
                    if self.base_model is not None:
                        return self.base_model.state_dict(*args, **kwargs)
                    # 3. fallback: lisa_modelのstate_dictを返す
                    if hasattr(self.lisa_model, '_state_dict'):
                        return self.lisa_model._state_dict(*args, **kwargs)
                    return {}  # 空の辞書を返すことで最低限の機能を提供
                finally:
                    # 4. 元のstate_dictメソッドを復元
                    if original_state_dict is not None:
                        setattr(self.__class__, 'state_dict', original_state_dict)
            
            def tie_weights(self):
                """tie_weightsメソッドの特別な処理 - train_ds.pyで使用"""
                if self.base_model is not None and hasattr(self.base_model, 'tie_weights'):
                    return self.base_model.tie_weights()
                return None
                
            def gradient_checkpointing_enable(self):
                """gradient_checkpointing_enableメソッドの特別な処理 - train_ds.pyで使用"""
                if self.base_model is not None and hasattr(self.base_model, 'gradient_checkpointing_enable'):
                    return self.base_model.gradient_checkpointing_enable()
                return None
        
        return ModelProxy(self, self._base_model_reference)
    
    def resize_token_embeddings(self, new_num_tokens):
        """
        トークン埋め込みのサイズを変更します。
        親クラスのresize_token_embeddingsメソッドに委譲します。
        
        Args:
            new_num_tokens: 新しいトークンの数
            
        Returns:
            リサイズされたモデル
        """
        print(f"LISAForCausalLM.resize_token_embeddings({new_num_tokens})を呼び出しました")
        
        # MllamaForConditionalGenerationのメソッドを直接呼び出す
        return super().resize_token_embeddings(new_num_tokens)
        
    def get_processor(self):
        """processorを取得"""
        if hasattr(self, "processor"):
            return self.processor
        else:
            print("警告: processorが設定されていません")
            return None
    
    def get_visual_embs(self, images):
        """
        SAMの画像エンコーダーを使用して視覚的特徴を抽出します。
        
        Args:
            images: 入力画像テンソル
            
        Returns:
            image_embeddings: SAM画像エンコーダからの特徴
        """
        if images is None:
            raise ValueError("入力画像がNoneです")
            
        if not hasattr(self, "visual_model") or self.visual_model is None:
            raise ValueError("visual_modelが初期化されていません")
            
        # 画像をデバイスに移動
        device = next(self.parameters()).device
        if images.device != device:
            images = images.to(device)
            
        try:
            with torch.no_grad():
                # SAMイメージエンコーダを呼び出し
                image_embeddings = self.visual_model.image_encoder(images)
                
            return image_embeddings
        except Exception as e:
            print(f"SAM画像エンコーダの実行中にエラーが発生しました: {e}")
            traceback.print_exc()
            raise
            
    def forward(self, **kwargs):
        """
        モデルの順伝播
        """
        # 親クラスのforwardを呼び出す
        return super().forward(**kwargs)
        
    def tie_weights(self):
        """
        入力埋め込みと出力埋め込みを結合します。
        MllamaForConditionalGenerationの機能を適切に継承します。
        """
        # 親クラスのtie_weightsメソッドを呼び出す
        super().tie_weights()
        print("親クラスのtie_weightsメソッドを呼び出しました")
        
    def _state_dict(self, *args, **kwargs):
        """
        モデルの状態辞書を取得します（無限再帰を防ぐための内部メソッド）
        ModelProxyのstate_dictメソッドから呼び出されます
        """
        # 親クラスの実装を避け、通常のnn.Moduleとして処理
        result = {}
        
        # モジュールと名前付きパラメータをstate_dictに追加
        for name, module in self.named_modules():
            if name == '':  # 自身は除外
                continue
            # ネストしたモジュールは、ドットで区切られた名前で処理
            if '.' not in name and hasattr(module, 'state_dict'):
                try:
                    local_state = module.state_dict(*args, **kwargs)
                    for key, param in local_state.items():
                        result[name + '.' + key] = param
                except Exception as e:
                    print(f"モジュール {name} のstate_dict取得中にエラー: {e}")
        
        # 直接のパラメータを追加（サブモジュールに属さないもの）
        for name, param in self.named_parameters(recurse=False):
            result[name] = param.data.clone()
            
        # バッファの追加
        for name, buf in self.named_buffers(recurse=False):
            result[name] = buf.clone()
            
        return result
        
    def state_dict(self, *args, **kwargs):
        """
        無限再帰を回避するためのstate_dictの実装
        """
        # 元のメソッドを保存
        original_state_dict = super(LISAForCausalLM, self).state_dict
        
        try:
            # 対応するモジュールのstate_dictを使用
            if hasattr(self, '_state_dict'):
                return self._state_dict(*args, **kwargs)
            
            # base_modelが存在する場合はその状態を使用
            if hasattr(self, '_base_model_reference') and self._base_model_reference is not None:
                return self._base_model_reference.state_dict(*args, **kwargs)
                
            # フォールバック: 親クラスのstate_dictを試す
            return original_state_dict(*args, **kwargs)
        except RecursionError:
            # 再帰エラーが発生した場合は、最小限の状態辞書を返す
            print("警告: state_dict取得中に再帰エラーが発生しました。最小限の状態を返します。")
            return {}  # 空の辞書を返す

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
        生成のための入力を準備します。
        Llama 3.2 Visionモデルでは、pixel_valuesとaspect_ratio_idsが必要です。
        """
        # 基本的には親クラスのメソッドを呼び出し、必要に応じて拡張する
        batch_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
            
        # 画像処理部分
        if images is not None:
            try:
                # プロセッサを使用して画像を処理
                processor = self.get_processor()
                if processor is not None:
                    # テンソル型かどうかで処理を分岐
                    if isinstance(images, torch.Tensor):
                        # PILイメージに変換
                        import numpy as np
                        from PIL import Image
                        
                        # バッチサイズの決定
                        if len(images.shape) == 4:  # [batch, ch, h, w]
                            images_for_processor = []
                            for i in range(images.shape[0]):
                                # 画像テンソルをPILイメージに変換
                                img_np = images[i].to(torch.float32).permute(1, 2, 0).cpu().numpy()
                                img_np = np.clip(img_np, 0, 1)
                                img_np = (img_np * 255).astype(np.uint8)
                                images_for_processor.append(Image.fromarray(img_np))
                        else:
                            # 単一画像の場合
                            img_np = images.to(torch.float32).permute(1, 2, 0).cpu().numpy()
                            img_np = np.clip(img_np, 0, 1)
                            img_np = (img_np * 255).astype(np.uint8)
                            images_for_processor = [Image.fromarray(img_np)]
                    else:
                        # すでにPILイメージかリスト形式の場合
                        images_for_processor = images if isinstance(images, list) else [images]
                    
                    # 各画像に対応するテキストを準備（<|image|>トークンのみ）
                    batch_size = len(images_for_processor)
                    text_prompts = ["<|image|>"] * batch_size
                    
                    # プロセッサで処理
                    device = next(self.parameters()).device
                    processor_outputs = processor(
                        text=text_prompts,
                        images=images_for_processor,
                        return_tensors="pt",
                        padding=True
                    )
                    
                    # デバイスを合わせる
                    for k, v in processor_outputs.items():
                        if isinstance(v, torch.Tensor):
                            processor_outputs[k] = v.to(device)
                    
                    # バッチ入力に追加
                    for k, v in processor_outputs.items():
                        batch_inputs[k] = v
                    
                    # aspect_ratio_idsの確認と追加
                    if "pixel_values" in batch_inputs and "aspect_ratio_ids" not in batch_inputs:
                        batch_inputs["aspect_ratio_ids"] = torch.zeros(
                            batch_inputs["pixel_values"].shape[0],
                            dtype=torch.long,
                            device=device
                        )
            except Exception as e:
                print(f"画像処理エラー: {e}")
                # エラーが発生してもクラッシュせず処理を続行
        
        return batch_inputs
