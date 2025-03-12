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

from .llama3_2.model.language_model.llama3_2 import (Llama3VisionForCausalLM,
                                                  Llama3VisionMetaModel)
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
            config.train_mask_decoder = kwargs.get("train_mask_decoder", True)
            config.out_dim = kwargs.get("out_dim", 256)
            config.vision_tower = kwargs.get("vision_tower", "meta-llama/Llama-3.2-11B-Vision-Instruct")
        
        # vision_towerをmodel_nameとして渡す
        model_name = kwargs.get("vision_tower") or getattr(config, "vision_tower", None)
        if model_name is None:
            model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        
        print("LisaModel初期化: 継承順序デバッグ開始")
        print(f"LisaModel初期化前 - クラスのMRO (Method Resolution Order): {[cls.__name__ for cls in self.__class__.__mro__]}")
        
        # LisaMetaModelを先に初期化して必要なフィールドを設定
        print("LisaMetaModel初期化前...")
        LisaMetaModel.__init__(self, config, **kwargs)
        print("LisaMetaModel初期化後...")
        
        # visual_modelの存在を確認
        if hasattr(self, 'visual_model'):
            print("LisaMetaModel初期化後: visual_model属性が存在します")
            print(f"  - visual_model type: {type(self.visual_model).__name__}")
            # 念のため、visual_modelを一時変数に保存
            original_visual_model = self.visual_model
        else:
            print("警告: LisaMetaModel初期化後、visual_model属性が見つかりません")
            original_visual_model = None
        
        # 次にLlama3VisionMetaModelを初期化
        print("Llama3VisionMetaModel初期化前...")
        Llama3VisionMetaModel.__init__(self, config, model_name=model_name, **kwargs)
        print("Llama3VisionMetaModel初期化後...")
        
        # 継承後のvisual_modelの状態を確認
        if hasattr(self, 'visual_model'):
            print("継承完了後: visual_model属性が存在します")
            print(f"  - visual_model type: {type(self.visual_model).__name__}")
        else:
            print("警告: 継承完了後、visual_model属性が失われています")
            
        # original_visual_modelがあり、現在のvisual_modelと異なる場合や存在しない場合は復元
        if original_visual_model is not None:
            if not hasattr(self, 'visual_model') or self.visual_model is not original_visual_model:
                print("visual_model属性を復元します")
                self.visual_model = original_visual_model
        
        # 属性のリストを出力
        print("LisaModel初期化後の全属性リスト:")
        obj_attrs = [attr for attr in dir(self) if not attr.startswith('__')]
        print(f"  - 属性: {', '.join(obj_attrs[:20])}...")
        
        # 重要: visual_modelが多重継承で上書きされている可能性があるため、
        # LisaMetaModelのvisual_modelを明示的に参照して保持する
        # これによりLISAForCausalLMからも正しく参照できるようになる
        if not hasattr(self, 'visual_model'):
            print("警告: visual_model属性が継承後に失われています。明示的に再設定します。")
            # SAMモデルを再初期化
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


class LISAForCausalLM(nn.Module):
    def __init__(
        self,
        model_id=None,
        model=None,
        config=None,
        cache_dir=None,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
        train_mask_decoder=True,
        out_dim=256,
        ce_loss_weight=1.0,
        dice_loss_weight=0.5,
        bce_loss_weight=2.0,
        seg_token_idx=0,
        vision_pretrained="PATH/TO/SAM/CHECKPOINT",
        vision_tower="openai/clip-vit-large-patch14",
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
            
        # tryブロックの外でLISAモデルを初期化
        # 必要なパラメータを準備
        lisa_kwargs = {
            'train_mask_decoder': train_mask_decoder,
            'out_dim': out_dim,
            'vision_pretrained': vision_pretrained,
            'vision_tower': vision_tower,
            'initialize_sam': True  # SAMの初期化を明示的に実行
        }
        
        # configがNoneの場合、モデルの設定を使用して新しいConfigオブジェクトを作成
        if config is None:
            if hasattr(self.model, 'config'):
                # モデルの設定を基にconfigを作成
                config = self.model.config
                # 必要な属性を追加
                if not hasattr(config, "train_mask_decoder"):
                    config.train_mask_decoder = train_mask_decoder
                if not hasattr(config, "out_dim"):
                    config.out_dim = out_dim
                print(f"モデル設定からconfigを生成: {type(config).__name__}")
            else:
                # 最低限必要な属性を持つ簡易configオブジェクトを作成
                from types import SimpleNamespace
                config = SimpleNamespace()
                config.train_mask_decoder = train_mask_decoder
                config.out_dim = out_dim
                config.vision_tower = vision_tower
                # MllamaConfigのように.text_configを持つ可能性を考慮
                if hasattr(self.model, 'config') and hasattr(self.model.config, 'text_config'):
                    config.text_config = self.model.config.text_config
                print("警告: 簡易configオブジェクトを生成しました")
        
        # LISAモデルの初期化
        self.lisa_model = LisaModel(config, **lisa_kwargs)
        
        # LISAモデルの視覚モデルを共有
        self.visual_model = self.lisa_model.visual_model

    def get_processor(self):
        """
        LISAモデル用のプロセッサを取得します
        """
        if not hasattr(self, "processor") or self.processor is None:
            self.processor = super().get_processor()
        return self.processor
    
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
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        **kwargs,
    ):
        # SAM用の特徴抽出
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        # セグメンテーショントークンのマスクを作成
        # 入力IDsの<SEG>トークンの位置を特定
        seg_token_mask = labels == self.seg_token_idx
        
        # Llama3.2 Visionモデルでは入力と出力の形式が異なるため、
        # <SEG>トークンの位置を特定するために適切なマスクを作成
        # プロセッサーを取得
        processor = self.get_processor()

        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            
            # 推論時は画像を拡張
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                
                # Llama3.2 vision用に入力を準備
                batch_inputs = processor(
                    text=input_ids[start_i:end_i],
                    images=images_clip_extend[: end_i - start_i],
                    return_tensors="pt",
                    padding=True,
                )
                
                # デバイスを合わせる
                batch_inputs = {k: v.to(self.model.device) for k, v in batch_inputs.items()}
                
                # モデルを実行
                output_i = self.model(**batch_inputs, output_hidden_states=True)
                output_hidden_states.append(output_i.hidden_states[-1])
                torch.cuda.empty_cache()

            # 出力を結合
            output_hidden_states = torch.cat(output_hidden_states, dim=0)
            output = None

        else:
            # 訓練時は各バッチごとに画像を拡張
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)
            
            # Llama3.2 vision用に入力を準備
            batch_inputs = processor(
                text=input_ids,
                images=images_clip,
                return_tensors="pt",
                padding=True,
            )
            
            # ラベルを設定
            if labels is not None:
                batch_inputs["labels"] = labels
                
            # デバイスを合わせる
            batch_inputs = {k: v.to(self.model.device) for k, v in batch_inputs.items()}
            
            # モデルを実行
            output = self.model(**batch_inputs, output_hidden_states=True)
            output_hidden_states = output.hidden_states[-1]

        # 以下はオリジナルのLISAと同様の処理
        hidden_states = []

        assert len(self.lisa_model.text_hidden_fcs) == 1
        hidden_states.append(self.lisa_model.text_hidden_fcs[0](output_hidden_states))

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        
        # <SEG>トークンの位置を特定
        seg_token_mask = labels == self.seg_token_idx
        
        # <SEG>トークンのembeddingを抽出
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )

        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

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
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])

        model_output = output
        gt_masks = masks_list

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        # ロス計算
        output = model_output.logits
        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

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
            # Llama3.2 vision用に入力を準備
            processor = self.get_processor()
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
            batch_inputs["images"] = images
            
        return batch_inputs
