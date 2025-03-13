from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BitsAndBytesConfig, 
    AutoProcessor, 
    AutoModelForVision2Seq, 
    AutoConfig, 
    GenerationMixin,
    MllamaForConditionalGeneration
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


class LisaModel(LisaMetaModel, Llama3VisionMetaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        # 事前にconfigの存在を確認
        if config is None:
            print("警告: LisaModelの初期化時にconfigがNoneです")
            from types import SimpleNamespace
            config = SimpleNamespace()
            
            # 必要な設定を追加
            for key, value in kwargs.items():
                setattr(config, key, value)
                
        # 親クラスの初期化 - 多重継承なので明示的に両方呼び出す
        print("LisaModelの初期化: 親クラスを初期化します")
        LisaMetaModel.__init__(self, config, **kwargs)
        Llama3VisionMetaModel.__init__(self, config, **kwargs)
        
        # SAMモデルが正しく初期化されたか確認
        if not hasattr(self, 'visual_model'):
            print("警告: visual_model属性が見つかりません。SAMモデルを初期化します。")
            # SAMモデルを初期化
            vision_pretrained = kwargs.get("vision_pretrained", None)
            if vision_pretrained:
                from .segment_anything import build_sam_vit_h
                self.visual_model = build_sam_vit_h(vision_pretrained)
                for param in self.visual_model.parameters():
                    param.requires_grad = False
                if config.train_mask_decoder:
                    self.visual_model.mask_decoder.train()
                    for param in self.visual_model.mask_decoder.parameters():
                        param.requires_grad = True
            else:
                print("エラー: vision_pretrainedが指定されていないため、SAMモデルを初期化できません")
                raise ValueError("SAMモデルの初期化に必要なvision_pretrainedパスが指定されていません")

        # 設定を構成
        self.config.use_cache = False
        
        # MllamaConfigではmm_接頭辞がない可能性がある属性の対応
        # vision_tower
        if hasattr(self.config, "mm_vision_tower"):
            self.config.vision_tower = self.config.mm_vision_tower
        # 既にvision_towerが設定されている場合は何もしない（MllamaConfigの場合）
        
        # vision_select_feature
        if not hasattr(self.config, "mm_vision_select_feature"):
            # MllamaConfig用に新しく属性を追加
            self.config.mm_vision_select_feature = "patch"
        
        # 他の設定属性を確実に設定
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False
        
        print("LisaModelの初期化: 成功しました")


class LISAForCausalLM(nn.Module):
    """
    LISA model for Causal Language Modeling with Segment Anything
    
    Llama3.2 Vision (MllamaForConditionalGeneration) + SAM
    """
    def __init__(
        self,
        model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        train_mask_decoder=True,
        out_dim=256,
        ce_loss_weight=1.0,
        dice_loss_weight=1.0,
        bce_loss_weight=1.0,
        seg_token_idx=None,
        vision_pretrained=None,
        vision_tower=None,
        use_mm_start_end=True,
        device_map=None,
    ):
        super().__init__()
        
        # LISA構成設定
        self.ce_loss_weight = ce_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.bce_loss_weight = bce_loss_weight
        self.train_mask_decoder = train_mask_decoder
        
        try:
            # モデルの初期化
            print(f"Loading Llama3.2 Vision model: {model_id}")
            print(f"  - torch_dtype: {torch_dtype}")
            
            # DeepSpeed環境ではdevice_mapをNoneに設定する必要がある
            # 'meta'デバイスのテンソルはDeepSpeedの初期化時にエラーが発生するため
            if device_map is not None:
                print(f"  - 警告: DeepSpeed環境ではdevice_map={device_map}を使用できません")
                print(f"  - device_map=Noneに設定します（DeepSpeedが自動的にデバイスを管理）")
                device_map = None
            
            print(f"  - device_map: {device_map}")
            
            # Llama3.2 Visionモデルのロード
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
                device_map=device_map
            )
            
            # config属性を追加（train_ds.pyでアクセスするため）
            self.config = self.model.config
            
            # PEFT互換性のために重要: model_typeがdictではなくConfigオブジェクトであることを確認
            # configが辞書の場合は、PretrainedConfigオブジェクトに変換
            if isinstance(self.model.config, dict):
                print("警告: configが辞書型です。PretrainedConfigオブジェクトに変換します。")
                from transformers import PretrainedConfig
                config_dict = self.model.config.copy()
                self.model.config = PretrainedConfig()
                for key, value in config_dict.items():
                    setattr(self.model.config, key, value)
            
            # text_configやvision_configなどのサブ設定も変換
            # Llama3.2 Visionモデルではこれらのサブ設定がネストされていることがある
            for config_name in ['text_config', 'vision_config']:
                if hasattr(self.model.config, config_name):
                    sub_config = getattr(self.model.config, config_name)
                    if isinstance(sub_config, dict):
                        print(f"警告: {config_name}が辞書型です。PretrainedConfigオブジェクトに変換します。")
                        sub_config_obj = PretrainedConfig()
                        for key, value in sub_config.items():
                            setattr(sub_config_obj, key, value)
                        setattr(self.model.config, config_name, sub_config_obj)
            
            # model_type属性が必要（PEFT用）
            if not hasattr(self.model.config, 'model_type'):
                print("configにmodel_type属性を追加します（PEFT用）")
                self.model.config.model_type = "mllama"
            
            # サブ設定にもmodel_type属性を追加
            for config_name in ['text_config', 'vision_config']:
                if hasattr(self.model.config, config_name):
                    sub_config = getattr(self.model.config, config_name)
                    if hasattr(sub_config, 'model_type') and sub_config.model_type is None:
                        if config_name == 'text_config':
                            sub_config.model_type = "llama"
                        elif config_name == 'vision_config':
                            sub_config.model_type = "vision_encoder"
                    elif not hasattr(sub_config, 'model_type'):
                        if config_name == 'text_config':
                            sub_config.model_type = "llama"
                        elif config_name == 'vision_config':
                            sub_config.model_type = "vision_encoder"
            
            # Llama3.2 Vision用のプロセッサを初期化
            self.processor = AutoProcessor.from_pretrained(model_id)
            
            # <SEG>トークンを追加
            special_tokens = {"additional_special_tokens": ["<SEG>"]}
            num_added_tokens = self.processor.tokenizer.add_special_tokens(special_tokens)
            print(f"  - Added {num_added_tokens} special tokens: <SEG>")
            
            # トークナイザでSEGトークンのインデックスを保存
            self.seg_token_idx = self.processor.tokenizer.convert_tokens_to_ids("<SEG>")
            print(f"  - <SEG> token index: {self.seg_token_idx}")
            
            # 埋め込みをリサイズ
            # 入力埋め込みのリサイズ
            # MllamaConfig対応: text_config.vocab_sizeから取得
            if hasattr(self.model.config, 'vocab_size'):
                orig_num_tokens = self.model.config.vocab_size
            elif hasattr(self.model.config, 'text_config') and hasattr(self.model.config.text_config, 'vocab_size'):
                # MllamaConfigではtext_config内に語彙サイズがある
                orig_num_tokens = self.model.config.text_config.vocab_size
                # 後続の処理で参照されるように設定
                self.model.config.vocab_size = orig_num_tokens
                print(f"  - MllamaConfig: text_config.vocab_sizeから語彙サイズを設定 ({orig_num_tokens})")
            else:
                # 最終手段: トークナイザーから直接サイズを取得
                orig_num_tokens = len(self.processor.tokenizer) - num_added_tokens
                self.model.config.vocab_size = orig_num_tokens
                print(f"  - 警告: configにvocab_sizeがないため、トークナイザーから推定 ({orig_num_tokens})")
            
            new_num_tokens = len(self.processor.tokenizer)
            print(f"  - Resizing embeddings from {orig_num_tokens} to {new_num_tokens}")
            
            # 入力埋め込みをリサイズ
            self.model.resize_token_embeddings(new_num_tokens)
            
            # 出力埋め込みのリサイズ（手動）
            try:
                output_embeddings = self.model.get_output_embeddings()
                
                # メタデバイスチェック
                if hasattr(output_embeddings, 'weight') and output_embeddings.weight.device.type == 'meta':
                    print("警告: メタデバイス上の出力埋め込みを検出しました")
                    print("DeepSpeed環境では通常の方法でリサイズできません")
                    print("入力埋め込みのリサイズのみ完了。後で重み共有を行います")
                else:
                    # 通常のケース - 出力埋め込みが実デバイス上にある場合
                    try:
                        # まず_get_resized_lm_headメソッドを使用してみる (推奨アプローチ)
                        if hasattr(self.model, '_get_resized_lm_head'):
                            print("  - _get_resized_lm_headメソッドを使用して出力埋め込みをリサイズ")
                            
                            # デバッグ: _get_resized_lm_headメソッドの引数を確認
                            import inspect
                            if hasattr(self.model, '_get_resized_lm_head'):
                                print("  - _get_resized_lm_head メソッドの引数情報:")
                                sig = inspect.signature(self.model._get_resized_lm_head)
                                print(f"    引数リスト: {list(sig.parameters.keys())}")
                                print(f"    デフォルト値: {[p.default for p in sig.parameters.values() if p.default is not inspect.Parameter.empty]}")
                            
                            # mean_resizingが引数に含まれているか確認して条件分岐
                            if 'mean_resizing' in inspect.signature(self.model._get_resized_lm_head).parameters:
                                new_output_embeddings = self.model._get_resized_lm_head(
                                    output_embeddings,
                                    new_num_tokens=new_num_tokens,
                                    mean_resizing=True
                                )
                            else:
                                # mean_resizingがない場合は引数なしで呼び出し
                                print("  - mean_resizing引数なしで_get_resized_lm_headを呼び出します")
                                new_output_embeddings = self.model._get_resized_lm_head(
                                    output_embeddings,
                                    new_num_tokens=new_num_tokens
                                )
                            
                            # 勾配設定を元に戻す
                            new_output_embeddings.requires_grad_(output_embeddings.weight.requires_grad)
                        else:
                            # フォールバック: 手動で出力埋め込みをリサイズ
                            print("  - 手動で出力埋め込みをリサイズ")
                            new_output_embeddings = torch.nn.Linear(
                                output_embeddings.in_features,
                                new_num_tokens,
                                bias=output_embeddings.bias is not None,
                                device=output_embeddings.weight.device
                            )
                            
                            with torch.no_grad():
                                # 既存のトークンの埋め込みをコピー
                                new_output_embeddings.weight.data[:orig_num_tokens, :] = output_embeddings.weight.data
                                # 新しいトークンの埋め込みを小さな乱数で初期化
                                new_output_embeddings.weight.data[orig_num_tokens:, :].normal_(mean=0.0, std=0.02)
                                
                                if output_embeddings.bias is not None:
                                    new_output_embeddings.bias.data[:orig_num_tokens] = output_embeddings.bias.data
                                    new_output_embeddings.bias.data[orig_num_tokens:] = 0
                        
                        # 新しい出力埋め込みを設定
                        self.model.set_output_embeddings(new_output_embeddings)
                        
                        # 重み共有設定の確認
                        if hasattr(self.model.config, 'tie_word_embeddings') and self.model.config.tie_word_embeddings:
                            # 重み共有が有効な場合のみtie_weightsを実行
                            print("入力/出力埋め込みの重みを共有（タイying）します")
                            self.model.tie_weights()
                        else:
                            # Llama 3.2などの非共有モデルの場合
                            print("このモデルは入出力埋め込み非共有モデル（tie_word_embeddings=False）です")
                            print("出力埋め込みは手動で初期化されました")
                        
                    except Exception as e:
                        print(f"警告: 出力埋め込みの初期化中にエラーが発生しました: {e}")
                        print("DeepSpeed環境での処理中は正常なため、続行します")
                
            except Exception as e:
                print(f"警告: 出力埋め込みの初期化中にエラーが発生しました: {e}")
                print("DeepSpeed環境での処理中は正常なため、続行します")
            
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise
            
        # 重要: モデル初期化後にLISA用の設定を行う
        lisa_config = {
            'train_mask_decoder': train_mask_decoder,
            'out_dim': out_dim,
            'vision_pretrained': vision_pretrained,
            'vision_tower': vision_tower,
            'initialize_sam': True  # SAMの初期化を明示的に実行
        }
        
        # configがNoneの場合、モデルの設定を使用して新しいConfigオブジェクトを作成
        # まずconfigを初期化する（model.configから取得）
        if hasattr(self.model, 'config'):
            config = self.model.config
            # 必要な属性を追加
            if not hasattr(config, "train_mask_decoder"):
                config.train_mask_decoder = train_mask_decoder
                config.out_dim = out_dim
                config.vision_pretrained = vision_pretrained
                config.vision_tower = vision_tower
        else:
            # モデルのconfigが存在しない場合、エラーを表示
            print("エラー: モデルにconfig属性がありません。LISAモデルを初期化できません。")
            raise ValueError("モデルの設定情報が見つかりません。")
            
        # Llama3VisionMetaModelとLisaMetaModelを継承したLisaModelを作成
        self.lisa_model = LisaModel(config, **lisa_config)
        
        # LISAモデルの視覚モデルを共有
        self.visual_model = self.lisa_model.visual_model

    def get_processor(self):
        """
        LISAモデル用のプロセッサを取得します
        """
        if not hasattr(self, "processor") or self.processor is None:
            self.processor = super().get_processor()
        return self.processor
    
    def get_model(self):
        """
        内部モデルを返します。
        train_ds.py内でmodel.model.configにアクセスするために必要です。
        """
        return self.model
    
    def get_input_embeddings(self):
        """
        入力埋め込み層を返します。
        PEFTのLoRAを適用するために必要です。
        """
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings()
        # モデルが直接メソッドを持っていない場合は、埋め込み層を直接取得
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            return self.model.model.embed_tokens
        # MllamaモデルではLlamaモデル部分の埋め込み層を取得
        if hasattr(self.model, "text_model") and hasattr(self.model.text_model, "embed_tokens"):
            return self.model.text_model.embed_tokens
        # 最後の手段として例外をスロー
        raise NotImplementedError("このモデルでは入力埋め込み層が見つかりません")
    
    def get_output_embeddings(self):
        """
        出力埋め込み層を返します。
        PEFTの一部の操作に必要です。
        """
        if hasattr(self.model, "get_output_embeddings"):
            return self.model.get_output_embeddings()
        # モデルが直接メソッドを持っていない場合は、出力埋め込み層を直接取得
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        # MllamaモデルではLlamaモデル部分の出力埋め込み層を取得
        if hasattr(self.model, "text_model") and hasattr(self.model.text_model, "lm_head"):
            return self.model.text_model.lm_head
        # 最後の手段として例外をスロー
        raise NotImplementedError("このモデルでは出力埋め込み層が見つかりません")
    
    def set_input_embeddings(self, value):
        """
        入力埋め込み層を設定します。
        resize_token_embeddingsで必要です。
        """
        if hasattr(self.model, "set_input_embeddings"):
            return self.model.set_input_embeddings(value)
        # モデルが直接メソッドを持っていない場合は、埋め込み層を直接設定
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            self.model.model.embed_tokens = value
            return
        # MllamaモデルではLlamaモデル部分の埋め込み層を設定
        if hasattr(self.model, "text_model") and hasattr(self.model.text_model, "embed_tokens"):
            self.model.text_model.embed_tokens = value
            return
        # 最後の手段として例外をスロー
        raise NotImplementedError("このモデルでは入力埋め込み層を設定できません")
    
    def set_output_embeddings(self, value):
        """
        出力埋め込み層を設定します。
        resize_token_embeddingsで必要です。
        """
        if hasattr(self.model, "set_output_embeddings"):
            return self.model.set_output_embeddings(value)
        # モデルが直接メソッドを持っていない場合は、出力埋め込み層を直接設定
        if hasattr(self.model, "lm_head"):
            self.model.lm_head = value
            return
        # MllamaモデルではLlamaモデル部分の出力埋め込み層を設定
        if hasattr(self.model, "text_model") and hasattr(self.model.text_model, "lm_head"):
            self.model.text_model.lm_head = value
            return
        # 最後の手段として例外をスロー
        raise NotImplementedError("このモデルでは出力埋め込み層を設定できません")
    
    def tie_weights(self):
        """
        入力埋め込みと出力埋め込みを結合します（同じパラメータを共有）。
        """
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is not None:
            self.set_output_embeddings(self.get_input_embeddings())
            
    def resize_token_embeddings(self, new_num_tokens=None):
        """
        モデルのトークン埋め込みのサイズを変更します。
        
        Args:
            new_num_tokens (int, optional): 新しいトークン数
            
        Returns:
            torch.nn.Embedding: 新しいトークン埋め込み層
        """
        # まず、モデルが直接resize_token_embeddingsを持っているか確認
        if hasattr(self.model, "resize_token_embeddings"):
            return self.model.resize_token_embeddings(new_num_tokens)
        
        # 内部実装：
        old_embeddings = self.get_input_embeddings()
        old_num_tokens = old_embeddings.num_embeddings
        
        if new_num_tokens == old_num_tokens:
            return old_embeddings
            
        # 新しい埋め込み層を作成
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        
        # モデルに新しい埋め込み層を設定
        self.set_input_embeddings(new_embeddings)
        
        # もし入力と出力の埋め込みが同じなら、出力も更新
        if self.get_output_embeddings() is not None and not getattr(self, "_tied_weights_keys", []):
            self.set_output_embeddings(new_embeddings)
            
        return new_embeddings
    
    def _get_resized_embeddings(self, old_embeddings, new_num_tokens=None):
        """
        リサイズされた埋め込み層を取得します。
        """
        if new_num_tokens is None:
            return old_embeddings
            
        old_num_tokens = old_embeddings.num_embeddings
        old_embedding_dim = old_embeddings.embedding_dim
        
        # 新しい埋め込み層を作成
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(old_embeddings.weight.device, 
                        dtype=old_embeddings.weight.dtype)
        
        # 既存の埋め込みをコピー
        with torch.no_grad():
            num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
            new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]
            
        return new_embeddings

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.lisa_model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            # Llama3.2 visionモデルの標準forward
            return self.model(**kwargs)
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
        inference=False,
        tokenizer=None,
        **kwargs
    ):
        """LISA/VLオブジェクトの前方伝播処理と損失計算を行う。
        
        注意: Llama 3.2 Visionモデル（MllamaForConditionalGeneration）では、
        画像入力は'images'ではなく'pixel_values'パラメータとして渡す必要があります。
        このメソッドでは内部的にimages_clipを'pixel_values'として渡します。
        
        理想的には以下の方法で入力を準備するとよいでしょう:
        ```
        processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")
        inputs = processor(text=text_prompt, images=image, return_tensors="pt")
        outputs = model(**inputs)
        ```
        
        processorは画像を適切な形式に変換し、<|image|>トークンを処理します。

        Args:
            input_ids (torch.Tensor): 入力トークンのID
            attention_mask (torch.Tensor, optional): アテンションマスク
            attention_masks (torch.Tensor, optional): 互換性のためのアテンションマスク
            labels (torch.Tensor, optional): ラベルデータ。デフォルトはNone。
            images (list, optional): 画像のリスト。デフォルトはNone。
            images_clip (list, optional): CLIP形式の画像のリスト。デフォルトはNone。
            masks_list (list, optional): マスクのリスト。デフォルトはNone。
            label_masks_list (list, optional): ラベルのマスクのリスト。デフォルトはNone。
            label_list (list, optional): 互換性のためのラベルマスクのリスト。
            inputs_embeds (torch.Tensor, optional): 入力埋め込み。デフォルトはNone。
            offset (torch.Tensor, optional): オフセット値。デフォルトはNone。
            resize_list (list, optional): リサイズリスト。デフォルトはNone。
            inference (bool, optional): 推論モードかどうか。デフォルトはFalse。

        Returns:
            dict: 計算された損失と出力値を含む辞書
        """
        # パラメータの互換性処理
        if attention_mask is None and attention_masks is not None:
            attention_mask = attention_masks
            
        if label_masks_list is None and label_list is not None:
            label_masks_list = label_list
            
        # オリジナルのLISAとの互換性のためにoffsetを処理
        batch_size = len(input_ids) if isinstance(input_ids, list) else input_ids.shape[0]
            
        # SEGトークンの埋め込みを取得
        with torch.no_grad():
            # self.deviceがないので、代わりにモデルの現在のデバイスを取得
            device = next(self.parameters()).device
            embedding_token_seg = self.get_input_embeddings()(torch.tensor([[self.seg_token_idx]], device=device))
            embedding_token_seg = embedding_token_seg.squeeze(0)
        
        # Llama 3.2 Visionを使用するためにprocessorで入力を準備
        # processorはpixel_valuesとaspect_ratio_idsを自動的に生成
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
            text_input = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
            print(f"デコードされたテキスト（デバッグ用）: {text_input[:2]}")  # 最初の2つのみ表示
        else:
            # すでに文字列または文字列のリストの場合はそのまま使用
            text_input = input_ids
            
        if not isinstance(text_input, (str, list)):
            # 文字列または文字列のリストでない場合はエラー
            raise ValueError(
                f"text_inputの型が無効です: {type(text_input)}。"
                "processorは文字列または文字列のリストを期待しています。"
            )
        
        # processorで入力を準備
        processor_inputs = processor(
            text=text_input,
            images=images_clip,
            return_tensors="pt",
            padding=True,
        )
        
        # デバイスを合わせる
        processor_inputs = {k: v.to(device) for k, v in processor_inputs.items()}
        
        # ラベルを追加（存在する場合）
        if labels is not None:
            processor_inputs["labels"] = labels
            
        # 出力のhidden statesを要求
        processor_inputs["output_hidden_states"] = True
        
        # 必要なパラメータでモデルを呼び出す
        outputs = self.model(**processor_inputs)
        
        # embeddings処理
        embeddings = torch.stack(outputs.hidden_states).squeeze(1)[-1]
        segmasks = []
        seg_token_counts = []
        seg_token_offset = []
        
        # SEGトークンを見つけるためのマスクを作成
        segment_token_mask = (input_ids == self.seg_token_idx)
        
        # 各サンプルでSEGトークンの位置と数を計算
        for b_idx in range(batch_size):
            seg_token_count = segment_token_mask[b_idx].sum().item()
            seg_token_counts.append(seg_token_count)
            seg_token_offset.append(len(segmasks))
            segmasks.extend([None] * seg_token_count)
        
        # SEGトークンの埋め込みを抽出
        segment_token_embedding_indices = segment_token_mask.nonzero().tolist()
        segment_token_embeddings = [embeddings[b_idx, s_idx, :].squeeze(0) for b_idx, s_idx in segment_token_embedding_indices]
        
        # 各SEGトークンの埋め込みから予測マスクを生成
        if len(segment_token_embedding_indices) > 0:
            # MLP投影
            segment_token_embeddings = torch.stack(segment_token_embeddings)
            segment_token_embeddings = self.token_embedding_projection(segment_token_embeddings)
            if self.vision_model is not None:
                # SAMのような視覚モデルを使用する場合
                for idx, (embedding, (b_idx, s_idx)) in enumerate(zip(segment_token_embeddings, segment_token_embedding_indices)):
                    if images is not None and b_idx < len(images):
                        image = images[b_idx]
                        predicted_mask = self.vision_model.predict_masks(
                            embedding.squeeze().unsqueeze(0),
                            input_images=image
                        ).squeeze()
                        # 低解像度マスクを処理（必要に応じて）
                        if predicted_mask.ndim == 3:
                            predicted_mask = predicted_mask.squeeze(0)
                            
                        # マスクを画像の元のサイズにリサイズ
                        if isinstance(image, torch.Tensor):
                            orig_size = image.shape[-2:]  # (H, W)
                        else:
                            orig_size = image.size[::-1]  # (W, H) -> (H, W)
                            
                        # リサイズ処理
                        if predicted_mask.shape != orig_size:
                            predicted_mask = F.interpolate(
                                predicted_mask.unsqueeze(0).unsqueeze(0), 
                                size=orig_size, 
                                mode='bilinear', 
                                align_corners=False
                            ).squeeze()
                            
                        # マスクを保存
                        segmasks[seg_token_offset[b_idx] + idx] = predicted_mask
        
        # 損失の計算
        ce_loss = None if labels is None else outputs.loss
        
        # マスク損失の計算準備
        mask_bce_loss = torch.tensor(0., device=device)
        mask_dice_loss = torch.tensor(0., device=device)
        num_masks = 0
        
        # 有効なマスクの数をカウント
        if masks_list is not None:
            valid_masks_count = sum(1 for masks in masks_list if masks is not None)
            if valid_masks_count == 0:
                print(f"WARNING: [マスクNull原因] すべてのマスクがNoneです。バッチサイズ: {batch_size}")
                # マスクがすべてNoneの場合はマスク損失を計算せずにテキスト生成損失のみ使用
                loss = ce_loss if ce_loss is not None else torch.tensor(0., device=device)
                return {"loss": loss, "ce_loss": ce_loss, "mask_dice_loss": mask_dice_loss, "mask_bce_loss": mask_bce_loss}
        
        # マスク損失の計算（マスクが存在する場合のみ）
        if label_masks_list is not None and masks_list is not None:
            for b_idx in range(batch_size):
                # マスクデータの取得
                gt_masks = masks_list[b_idx] if b_idx < len(masks_list) else None
                label_masks = label_masks_list[b_idx] if b_idx < len(label_masks_list) else None
                
                # デバッグ情報：マスクがNoneの場合の詳細情報
                if gt_masks is None:
                    print(f"WARNING: [マスクNull原因] バッチインデックス {b_idx} のマスクがNoneです")
                    continue
                    
                # デバッグ情報：ラベルマスクがNoneの場合の詳細情報
                if label_masks is None:
                    print(f"WARNING: [マスクNull原因] バッチインデックス {b_idx} のラベルマスクがNoneです")
                    continue
                
                # そのバッチのSEGトークン数を取得
                seg_count = seg_token_counts[b_idx]
                
                # SEGトークンが存在しない場合は損失に含めない
                if seg_count == 0:
                    continue
                
                # SEGトークンに対応するマスクが存在する場合
                if seg_count == 1 and len(gt_masks) == 1:
                    # 1対1のマッピング
                    gt_mask = gt_masks[0]
                    pred_mask_idx = seg_token_offset[b_idx]
                    
                    # 予測マスクを取得
                    pred_mask = segmasks[pred_mask_idx]
                    if pred_mask is None:
                        print(f"WARNING: [マスクNull原因] バッチインデックス {b_idx} の予測マスクがNoneです")
                        continue
                    
                    # マスク損失の計算
                    mask_bce_loss += F.binary_cross_entropy_with_logits(pred_mask, gt_mask)
                    mask_dice_loss += 1 - compute_dice_loss(pred_mask, gt_mask)
                    num_masks += 1
                    
                elif seg_count > 0 and len(gt_masks) > 0:
                    # 複数マスクの場合は最適なマッチングを見つける
                    best_match_cost = float('inf')
                    best_match_masks = None
                    
                    # 存在するすべての予測マスクを確認
                    pred_masks = []
                    for i in range(seg_count):
                        pred_mask_idx = seg_token_offset[b_idx] + i
                        pred_mask = segmasks[pred_mask_idx]
                        if pred_mask is not None:
                            pred_masks.append(pred_mask)
                    
                    if not pred_masks:
                        print(f"WARNING: [マスクNull原因] バッチインデックス {b_idx} には予測マスクがありません")
                        continue
                    
                    # データ型とデバイスを確認
                    gt_masks_tensor = torch.stack(gt_masks).to(device) if isinstance(gt_masks[0], torch.Tensor) else torch.tensor(gt_masks).to(device)
                    pred_masks_tensor = torch.stack(pred_masks)
                    
                    # サイズが一致することを確認
                    if gt_masks_tensor.shape[1:] != pred_masks_tensor.shape[1:]:
                        print(f"WARNING: [マスクNull原因] マスクサイズの不一致: gt_masks={gt_masks_tensor.shape}, pred_masks={pred_masks_tensor.shape}")
                        gt_masks_tensor = F.interpolate(
                            gt_masks_tensor.unsqueeze(1).float(), 
                            size=pred_masks_tensor.shape[1:], 
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze(1)
                    
                    # ハンガリアン法で最適マッチングを計算
                    from scipy.optimize import linear_sum_assignment
                    
                    cost_matrix = np.zeros((len(pred_masks), len(gt_masks)))
                    for i, pred in enumerate(pred_masks):
                        for j, gt in enumerate(gt_masks):
                            bce_loss = F.binary_cross_entropy_with_logits(pred, gt)
                            dice_loss = 1 - compute_dice_loss(pred, gt)
                            cost_matrix[i, j] = bce_loss.item() + dice_loss.item()
                    
                    # 最適割り当てを計算
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    
                    # 割り当てに基づいて損失を計算
                    for i, j in zip(row_ind, col_ind):
                        mask_bce_loss += F.binary_cross_entropy_with_logits(pred_masks[i], gt_masks[j])
                        mask_dice_loss += 1 - compute_dice_loss(pred_masks[i], gt_masks[j])
                        num_masks += 1
        
        # マスク損失の平均を計算
        if num_masks > 0:
            mask_bce_loss = mask_bce_loss / num_masks
            mask_dice_loss = mask_dice_loss / num_masks
            
            # 全体の損失を計算（テキスト生成とマスク損失）
            loss = ce_loss + self.mask_loss_weight * (mask_bce_loss + mask_dice_loss) if ce_loss is not None else self.mask_loss_weight * (mask_bce_loss + mask_dice_loss)
        else:
            print(f"WARNING: [マスクNull原因] 有効なマスクペアが見つかりませんでした")
            # マスクがない場合はテキスト生成損失のみを使用
            loss = ce_loss if ce_loss is not None else torch.tensor(0., device=device)
        
        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_bce_loss": mask_bce_loss
        }

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
        with torch.no_grad():
            # Llama3.2 vision用に入力を準備
            processor = self.get_processor()
            # processorは'images'を'pixel_values'に内部的に変換してくれる
            batch_inputs = processor(
                text=input_ids,
                images=images_clip,
                return_tensors="pt",
                padding=True,
            )
            
            # デバイスを合わせる
            batch_inputs = {k: v.to(self.model.device) for k, v in batch_inputs.items()}
            
            # 生成パラメータを設定
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "num_beams": 1,
                "output_hidden_states": True,
                "return_dict_in_generate": True,
            }
            
            # 生成を実行
            outputs = self.model.generate(**batch_inputs, **generation_config)
            
            # 出力を取得
            # 最後のレイヤーの隠れ状態を取得
            # Llama3.2では生成中に全レイヤーの隠れ状態を保存するが、
            # 必要なのは最終デコーダレイヤーの隠れ状態のみ
            output_hidden_states = outputs.hidden_states[-1][-1]  # 最後のトークンの最後のレイヤー
            output_ids = outputs.sequences
            
            # 生成されたテキストを表示（デバッグ用）
            if tokenizer:
                print("生成されたテキスト:", tokenizer.batch_decode(output_ids, skip_special_tokens=False))

            # <SEG>トークンの位置を特定
            # 生成された出力の中から<SEG>トークンの位置を検出
            seg_token_mask = output_ids == self.seg_token_idx

            # 以下はオリジナルのLISAと同様の処理
            hidden_states = []

            assert len(self.lisa_model.text_hidden_fcs) == 1
            hidden_states.append(self.lisa_model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            
            # <SEG>トークンの位置が見つからない場合の処理
            if not seg_token_mask.any():
                print("警告: 生成されたテキストに<SEG>トークンが見つかりません。空のマスクを返します。")
                return {
                    "pred_masks": [torch.zeros_like(image[0]) for image in images],
                    "output_text": processor.tokenizer.batch_decode(output_ids, skip_special_tokens=False),
                }
                
            # <SEG>トークンのembeddingを抽出
            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.lisa_model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.lisa_model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.lisa_model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.lisa_model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks

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
            
        return batch_inputs

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
