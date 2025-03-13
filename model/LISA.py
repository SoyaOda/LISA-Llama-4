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
        # SAMの特徴を抽出
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        # <SEG>トークンのマスクを作成
        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().to(seg_token_mask.device),
            ],
            dim=1,
        )

        # <SEG>トークンの埋め込みを取得
        embedding_token_seg = getattr(
            self.model.llama_model.model.embed_tokens,
            "embedding_token_seg",
            None,
        )
        if embedding_token_seg is None:
            # もし埋め込みが初期化されていなければ新しく作成
            embedding_token_seg = self.model.llama_model.model.embed_tokens(
                torch.tensor([self.seg_token_idx], device=self.model.device)
            )
            embedding_token_seg = embedding_token_seg.squeeze(0)

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
            
            print(f"processorに渡す画像形状: {images_for_processor.shape}")
        else:
            images_for_processor = None
            print("画像データなし")
        
        # processorで入力を準備
        try:
            if images_for_processor is not None:
                processor_inputs = processor(
                    text=text_input,
                    images=images_for_processor,
                    return_tensors="pt",
                    padding=True,
                )
            else:
                processor_inputs = processor(
                    text=text_input,
                    return_tensors="pt",
                    padding=True,
                )
            
            # デバイスを合わせる
            if device is not None:
                processor_inputs = {k: v.to(device) for k, v in processor_inputs.items()}
            
            print("プロセッサ実行成功")
        except Exception as e:
            print(f"プロセッサエラー: {e}")
            raise e
        
        # 変換されたpixel_valuesとaspect_ratio_idsをデバッグ表示
        if "pixel_values" in processor_inputs:
            print(f"変換されたpixel_values形状: {processor_inputs['pixel_values'].shape}")
        if "aspect_ratio_ids" in processor_inputs:
            print(f"aspect_ratio_ids: {processor_inputs['aspect_ratio_ids']}")
            
        # 出力のhidden statesを要求
        processor_inputs["output_hidden_states"] = True
        
        # 必要なパラメータでモデルを呼び出す
        outputs = self.model(**processor_inputs)
        
        # 出力からhidden_statesを取得
        last_hidden_state = outputs.hidden_states[-1]
        
        # <SEG>トークンの位置を特定して処理
        # hidden_statesを処理してマスク生成に使用
        hidden_states = []
        hidden_states.append(
            getattr(self.model, "text_hidden_fcs", [lambda x: x])[0](last_hidden_state)
        )

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]
        
        # <SEG>トークンのカウントとオフセットを計算
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().to(seg_token_mask.device), seg_token_offset], dim=0
        )

        # オフセットに基づいてseg_token_offsetを調整
        seg_token_offset = seg_token_offset[offset]

        # 予測埋め込みを処理
        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        # 予測マスクを生成
        multimask_output = False
        pred_masks = []
        for i in range(len(pred_embeddings)):
            # SAMのプロンプトエンコーダを使用してスパース埋め込みを生成
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            
            # データ型を合わせる
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            
            # マスクデコーダを実行
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            
            # 予測マスクを後処理
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            
            pred_masks.append(pred_mask[:, 0])

        # 推論モードの場合は予測マスクを返す
        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": masks_list,
            }

        # 損失計算のためのモデル出力と正解マスク
        model_output = outputs
        gt_masks = masks_list

        # 言語モデルのCE損失を取得
        ce_loss = model_output.loss if hasattr(model_output, 'loss') else torch.tensor(0.0).to(device)
        ce_loss = ce_loss * self.ce_loss_weight
        
        # マスク損失（BCE, Dice）の初期化
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        
        # バッチ内の各サンプルに対して損失を計算
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            # マスクの形状を確認
            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            
            # BCE損失とDice損失を計算
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        # 損失を正規化
        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        # 最終的な損失
        loss = ce_loss + mask_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
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
            # テキスト入力の準備
            if isinstance(input_ids, str):
                # 文字列が直接渡された場合
                if '<|image|>' not in input_ids:
                    # <|image|>トークンがなければ追加
                    user_prefix = "User: "
                    if user_prefix in input_ids:
                        pos = input_ids.find(user_prefix) + len(user_prefix)
                        text_input = input_ids[:pos] + "<|image|> " + input_ids[pos:]
                    else:
                        text_input = "<|image|> " + input_ids
                else:
                    text_input = input_ids
            elif tokenizer is not None and isinstance(input_ids, torch.Tensor):
                # トークンIDからテキストにデコード
                decoded_text = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
                text_input = []
                for text in decoded_text:
                    if '<|image|>' not in text:
                        user_prefix = "User: "
                        if user_prefix in text:
                            pos = text.find(user_prefix) + len(user_prefix)
                            text = text[:pos] + "<|image|> " + text[pos:]
                        else:
                            text = "<|image|> " + text
                    text_input.append(text)
            else:
                # その他の場合はそのまま使用
                text_input = input_ids
            
            # Llama3.2 vision用に入力を準備
            processor = self.get_processor()
            
            # processorは'images'を'pixel_values'に内部的に変換してくれる
            batch_inputs = processor(
                text=text_input,
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
            
            # テキスト生成を実行
            outputs = self.model.generate(**batch_inputs, **generation_config)
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            # <SEG>トークンの位置をマスクで特定
            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            # プレフィックスに対応する部分（Llama3.2の場合は画像トークンの分）をパディング
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 255)).bool().to(self.model.device),
                    seg_token_mask,
                ],
                dim=1,
            )

            # hidden_statesを処理
            hidden_states = []
            hidden_states.append(
                getattr(self.model, "text_hidden_fcs", [lambda x: x])[0](output_hidden_states)
            )

            # hidden_statesを合成
            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            
            # <SEG>トークンに対応するhidden_stateを抽出
            pred_embeddings = last_hidden_state[seg_token_mask]

            # <SEG>トークンの数とオフセットを計算
            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().to(self.model.device), seg_token_offset], dim=0
            )

            # 予測埋め込みを処理
            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            # SAMの画像埋め込みを取得
            image_embeddings = self.get_visual_embs(images)

            # 各<SEG>トークンに対応するマスクを生成
            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                # スパース埋め込みと密埋め込みを取得
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )

                # スパース埋め込みのデータ型を合わせる
                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                
                # マスクを生成
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                
                # マスクを後処理
                pred_mask = self.model.visual_model.postprocess_masks(
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
            
            # バッチサイズを取得
            if isinstance(images, torch.Tensor):
                batch_size = images.shape[0]
                # aspect_ratio_idsを作成（デフォルトで0）
                device = images.device
                batch_inputs["aspect_ratio_ids"] = torch.zeros(batch_size, dtype=torch.long, device=device)

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
