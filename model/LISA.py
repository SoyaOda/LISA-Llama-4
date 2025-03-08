from typing import List, Optional, Union, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel, AutoModelForVision2Seq, AutoConfig, AutoProcessor

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .segment_anything import build_sam_vit_h, sam_model_registry
from .llama3_2.constants import SEG_TOKEN


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
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.mean(1).sum() / (num_masks + 1e-8)
    return loss


# 元のLisaMetaModelとLisaModelは、LLaVA依存のため使用しません
# LisaMetaModelとLisaModelクラスはコメントアウトしておきます

# class LisaMetaModel:
#     def __init__(
#         self,
#         config,
#         **kwargs,
#     ):
#         self.config = config
#         # サポートされていない機能の警告をオフにする
#         self.config._attn_implementation = None
#         self.initialize_lisa_modules(config)
#
#     def initialize_lisa_modules(self, config):
#         # SAM
#         if "use_mm_start_end" not in config.__dict__:
#             config.use_mm_start_end = False
#         if "vision_tower" not in config.__dict__:
#             config.vision_tower = None
#         if "seg_token_idx" not in config.__dict__:
#             config.seg_token_idx = None
#         if "vision_feature_select_strategy" not in config.__dict__:
#             config.vision_feature_select_strategy = None
#         if "vision_feature_layer" not in config.__dict__:
#             config.vision_feature_layer = None
#         if "train_mask_decoder" not in config.__dict__:
#             config.train_mask_decoder = False
#         if "out_dim" not in config.__dict__:
#             config.out_dim = 256
#
#         # SAM'S ViT-H 画像エンコーダをロード
#         vision_tower = kwargs.pop("vision_tower", None)
#         if vision_tower is None and getattr(config, "vision_tower", None) is not None:
#             vision_tower = config.vision_tower
#         self.visual_model = build_sam_vit_h(vision_tower)
#         self.config.vision_tower = vision_tower
#
#         # SAMのパラメータを凍結
#         for param in self.visual_model.parameters():
#             param.requires_grad = False
#         if config.train_mask_decoder:
#             # 訓練モードでは、マスクデコーダを訓練可能に設定
#             self.visual_model.mask_decoder.train()
#             for param in self.visual_model.mask_decoder.parameters():
#                 param.requires_grad = True
#
#         # セグメンテーションモジュール
#         # from .segment_anything.modeling import MaskDecoder
#         self.ce_loss_weight = kwargs.pop("ce_loss_weight", 1.0)
#         self.dice_loss_weight = kwargs.pop("dice_loss_weight", 0.5)
#
#         hidden_size = config.hidden_size
#         out_dim = config.out_dim
#         # LLMの隠れ状態をマスク埋め込み次元にプロジェクションするレイヤー
#         self.text_hidden_fcs = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(hidden_size, hidden_size),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(hidden_size, out_dim),
#                 nn.Dropout(0.0)
#             )
#         ])


# class LisaModel(LisaMetaModel, LlavaLlamaModel):
#     def __init__(
#         self,
#         config,
#         **kwargs,
#     ):
#         LlavaLlamaModel.__init__(self, config)
#         LisaMetaModel.__init__(
#             self,
#             config,
#             **kwargs,
#         )


# 元のLISAForCausalLMクラスも使用しないため、コメントアウト
# class LISAForCausalLM(LlavaLlamaForCausalLM):
#     def __init__(
#         self,
#         config,
#         **kwargs,
#     ):
#         super(LISAForCausalLM, self).__init__(config)
#         self.model = LisaModel(config, **kwargs)
#         self.ce_loss_weight = kwargs.pop("ce_loss_weight", 1.0)
#         self.dice_loss_weight = kwargs.pop("dice_loss_weight", 0.5)
#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
#         # セグメントトークンの設定
#         seg_token_idx = kwargs.pop("seg_token_idx", None)
#         if seg_token_idx is None and getattr(config, "seg_token_idx", None) is not None:
#             seg_token_idx = config.seg_token_idx
#         self.config.seg_token_idx = seg_token_idx
#         # 使用キャッシュをオフに
#         self.config.use_cache = False
#         # Model parallel
#         self.model_parallel = False
#         self.device_map = None
#         # Initialize weights and apply final processing
#         self.post_init()
#
#     def get_visual_embs(self, pixel_values: torch.FloatTensor):
#         """SAMの視覚エンコーダを使用して画像埋め込みを取得"""
#         with torch.no_grad():
#             # SAMのViT-Hモデルが画像を処理
#             visual_outputs = self.model.visual_model.image_encoder(pixel_values)
#         return visual_outputs
#
#     def forward(self, **kwargs):
#         return self.model_forward(**kwargs)
#
#     # 以下のメソッドは使用しないためコメントアウト
#     def model_forward(
#         self,
#         images: torch.FloatTensor,
#         images_clip: torch.FloatTensor,
#         input_ids: torch.LongTensor,
#         labels: torch.LongTensor,
#         attention_masks: torch.LongTensor,
#         offset: torch.LongTensor,
#         masks_list: List[torch.FloatTensor],
#         label_list: List[torch.Tensor],
#         resize_list: List[tuple],
#         inference: bool = False,
#         **kwargs,
#     ):
#         pass
#
#     def evaluate(
#         self,
#         images_clip,
#         images,
#         input_ids,
#         resize_list,
#         original_size_list,
#         max_new_tokens=32,
#         tokenizer=None,
#     ):
#         pass


# Llama3.2 Vision + SAM統合版（このクラスのみ残す）
class Llama32LISAForCausalLM(nn.Module):
    """
    Llama3.2 Vision 11B InstructモデルとSAMを統合したLISAモデル。
    基本的なマルチモーダルモデルをベースに、SAMの視覚エンコーダをビジョンタワーとして使用します。
    """
    def __init__(self, config, **kwargs):
        """
        Llama32LISAForCausalLMを初期化
        
        Args:
            config: モデル設定
            vision_pretrained: SAMビジョンモデルの事前学習済み重みへのパス
            train_mask_decoder: マスクデコーダをトレーニングするかどうか
            out_dim: 出力次元数
            seg_token_idx: セグメントトークンのインデックス
        """
        # 設定を格納
        super().__init__()
        self.config = config
        
        # キーワード引数からパラメータを取得
        self.vision_ckpt = kwargs.pop("vision_pretrained", None)
        self.train_mask_decoder = kwargs.pop("train_mask_decoder", False)
        self.out_dim = kwargs.pop("out_dim", 256)
        self.seg_token_idx = kwargs.pop("seg_token_idx", None)
        
        # モデルが適切にロードされたかのフラグ
        self.model = None
        self.visual_model = None
        
        # 基本モデルは外部から設定（from_vision_modelメソッドから）
        
        # モデル設定をロガーに出力
        print(f"初期化: SAM checkpoint={self.vision_ckpt}, train_mask_decoder={self.train_mask_decoder}, "
              f"out_dim={self.out_dim}, seg_token_idx={self.seg_token_idx}")
        
        # MllamaConfigの構造を特定するデバッグ情報を出力
        if hasattr(config, 'model_type'):
            print(f"モデルタイプ: {config.model_type}")
        
        # 設定の構造を確認
        if hasattr(config, 'text_config'):
            print(f"テキスト設定: {type(config.text_config).__name__}")
            if hasattr(config.text_config, 'hidden_size'):
                print(f"テキスト隠れ層サイズ: {config.text_config.hidden_size}")
        elif hasattr(config, 'hidden_size'):
            print(f"隠れ層サイズ（直接）: {config.hidden_size}")
        
        # ビジョン設定も確認
        if hasattr(config, 'vision_config'):
            print(f"ビジョン設定: {type(config.vision_config).__name__}")
            if hasattr(config.vision_config, 'hidden_size'):
                print(f"ビジョン隠れ層サイズ: {config.vision_config.hidden_size}")
        
        # SAMエンコーダをロード（パスが指定されている場合）
        if self.vision_ckpt:
            try:
                self.initialize_sam_encoder()
            except Exception as e:
                print(f"SAMエンコーダの初期化中にエラーが発生: {e}")
                print("SAMエンコーダの初期化はスキップされました。後で手動で initialize_sam_encoder() を呼び出してください。")
                import traceback
                traceback.print_exc()
    
    @classmethod
    def from_vision_model(cls, vision_model_id, seg_token_idx=None, vision_pretrained=None, train_mask_decoder=False, 
                         out_dim=256, tokenizer=None, torch_dtype=None, device_map=None, 
                         quantization_config=None, ignore_mismatched_sizes=False, load_in_8bit=False, 
                         load_in_4bit=False, offload_folder=None, **kwargs):
        """
        Llama3.2 Vision modelとSAMを統合したモデルを生成します。
        
        Args:
            vision_model_id: Llama3.2 VisionモデルのIDまたはパス
            seg_token_idx: <SEG>トークンのインデックス
            vision_pretrained: SAMモデルの事前学習済み重みへのパス
            train_mask_decoder: SAMのマスクデコーダーを訓練するかどうか
            out_dim: 出力次元数
            tokenizer: 使用するトークナイザー
            torch_dtype: モデルのデータ型
            device_map: デバイスマッピング
            quantization_config: 量子化設定
            ignore_mismatched_sizes: サイズの不一致を無視するかどうか
            load_in_8bit: 8ビット量子化を使用するかどうか
            load_in_4bit: 4ビット量子化を使用するかどうか
            offload_folder: オフロード用一時フォルダのパス
            **kwargs: その他の引数
        """
        print("Step 1: Loading base vision-language model...")
        # 最初にトークナイザーをロードし拡張する
        if tokenizer is None:
            tokenizer = AutoProcessor.from_pretrained(vision_model_id)
        
        # <SEG>トークンをトークナイザーに追加
        if hasattr(tokenizer, 'tokenizer'):
            # Processor with internal tokenizer (MllamaProcessor)
            if "<SEG>" not in tokenizer.tokenizer.get_vocab():
                print("Adding <SEG> token to tokenizer vocabulary")
                tokenizer.tokenizer.add_special_tokens({"additional_special_tokens": ["<SEG>"]})
        else:
            # Direct tokenizer
            if "<SEG>" not in tokenizer.get_vocab():
                print("Adding <SEG> token to tokenizer vocabulary")
                tokenizer.add_special_tokens({"additional_special_tokens": ["<SEG>"]})
        
        # Mllamaタイプを検出
        is_mllama = 'mllama' in vision_model_id.lower() or (hasattr(tokenizer, 'image_token'))
        if is_mllama:
            print("Llama 3.2 Vision (Mllama)モデルを検出しました")
            
            # 画像トークンの情報
            if hasattr(tokenizer, 'image_token'):
                image_token = tokenizer.image_token
                print(f"Using image token: {image_token}")
        
        # 量子化設定がない場合は作成
        if quantization_config is None and (load_in_8bit or load_in_4bit):
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        
        # デバイスマップがない場合は設定（メモリ効率のため）
        if device_map is None and torch.cuda.is_available():
            # 自動的にGPUとCPU間でレイヤーを配置
            device_map = "auto"
        
        # メモリ効率のためのオプション
        model_loading_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "quantization_config": quantization_config,
            "ignore_mismatched_sizes": ignore_mismatched_sizes,
        }
        
        # オフロードフォルダが指定されている場合は追加
        if offload_folder:
            model_loading_kwargs["offload_folder"] = offload_folder
            model_loading_kwargs["low_cpu_mem_usage"] = True
        
        # トークナイザーを拡張した後にモデルをロード
        try:
            # メモリクリーンアップ
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            # モデルをロード
            vision_model = AutoModelForVision2Seq.from_pretrained(
                vision_model_id,
                **model_loading_kwargs,
                **kwargs
            )
        except Exception as e:
            print(f"モデルロード中にエラーが発生: {e}")
            print("代替方法でロード試行...")
            
            # 単純化した設定でロード試行
            simplified_kwargs = {
                "torch_dtype": torch_dtype,
                "ignore_mismatched_sizes": ignore_mismatched_sizes
            }
            
            # deviceを指定しない方法でロード
            vision_model = AutoModelForVision2Seq.from_pretrained(
                vision_model_id,
                **simplified_kwargs,
                **kwargs
            )
        
        # SEGトークンのインデックスを取得
        if seg_token_idx is None:
            if hasattr(tokenizer, 'tokenizer'):
                seg_token_idx = tokenizer.tokenizer.get_vocab()["<SEG>"]
            else:
                seg_token_idx = tokenizer.get_vocab()["<SEG>"]
        
        # モデルの設定を取得
        config = vision_model.config
        
        # LISA modelを初期化
        print("Step 2: Customizing with SAM integration...")
        model = cls(
            config,
            vision_pretrained=vision_pretrained,
            train_mask_decoder=train_mask_decoder,
            out_dim=out_dim,
            seg_token_idx=seg_token_idx,
            **kwargs
        )
        
        # SAMエンコーダを初期化（GPUメモリの効率化のためtorch.no_gradを使用）
        with torch.no_grad():
            model.initialize_sam_encoder()
        
        # 基本モデルの重みを転送
        print("Step 3: Transferring model weights...")
        model.model = vision_model
        
        return model, tokenizer
    
    def resize_token_embeddings(self, new_num_tokens):
        """トークン埋め込みのサイズを変更する"""
        if hasattr(self, 'model') and self.model is not None:
            self.model.resize_token_embeddings(new_num_tokens)
            return self
        return None
        
    def initialize_sam_encoder(self):
        """SAMエンコーダを初期化"""
        if self.vision_ckpt:
            print(f"Initializing SAM encoder from {self.vision_ckpt}...")
            
            # メモリクリーンアップ
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            # SAMエンコーダを初期化
            self.visual_model = sam_model_registry["vit_h"](checkpoint=self.vision_ckpt)
            
            for param in self.visual_model.parameters():
                param.requires_grad = False
                
            # プロンプトエンコーダをトレーニングしない
            if not self.train_mask_decoder:
                for param in self.visual_model.mask_decoder.parameters():
                    param.requires_grad = False

            # プロジェクションレイヤーの作成
            # MllamaConfigの構造に合わせてhidden_sizeを取得する
            if hasattr(self.config, 'text_config') and hasattr(self.config.text_config, 'hidden_size'):
                hidden_size = self.config.text_config.hidden_size
            elif hasattr(self.config, 'hidden_size'):
                # 従来のモデル用の互換性
                hidden_size = self.config.hidden_size
            else:
                # デフォルト値（Llama 3.2のテキストモデルの標準サイズ）
                hidden_size = 4096
            
            print(f"Using hidden size: {hidden_size} for projection layers")
            self.mm_projector = nn.Linear(256, hidden_size)
            
            # メモリ使用量削減のため必要ない変数を削除
            if not self.train_mask_decoder and hasattr(self.visual_model, 'prompt_encoder'):
                # プロンプトエンコーダが必要ない場合は削除
                self.visual_model.prompt_encoder = None
                
            # モデルの非トレーニング部分をCPUに移動
            if torch.cuda.is_available() and hasattr(self, "device"):
                # 必要な場合のみGPUにロード
                # 画像処理直前にGPUに移動するため、ここではCPUに保持
                self.visual_model = self.visual_model.cpu()
            torch.cuda.empty_cache()
    
    @property
    def device(self):
        """モデルのデバイスを取得"""
        if hasattr(self, 'model') and self.model is not None:
            return next(self.model.parameters()).device
        return next(self.parameters()).device
        
    def forward(
        self,
        input_ids=None, 
        attention_mask=None, 
        pixel_values=None, 
        labels=None,
        images=None,  # 元のサイズの画像（SAM用）
        masks_list=None,  # 訓練用マスク
        label_list=None,  # マスクのラベル
        resize_list=None,  # リサイズ情報
        inference=False,  # 推論モード
        **kwargs
    ):
        """モデルのフォワードパス。画像入力が提供された場合、SAMエンコーダでエンコードします。"""
        # SAM ViT-H特徴量を初期化
        vision_hidden_states = None
        
        # 画像入力がある場合、SAMエンコーダでエンコード
        if pixel_values is not None and hasattr(self, 'visual_model'):
            batch_size = pixel_values.shape[0]
            img_embeds_list = []
            
            for i in range(batch_size):
                # 各画像をエンコード（特徴マップ [1, C, H, W] を出力）
                with torch.no_grad():  # 視覚エンコーダは凍結されている
                    img_embed = self.visual_model.image_encoder(pixel_values[i:i+1])
                img_embeds_list.append(img_embed)
                
            # すべての画像埋め込みを連結
            image_embeds = torch.cat(img_embeds_list, dim=0)  # 形状 [B, C, H, W]
            
            # 必要に応じて、クロスアテンション用に空間次元をパッチトークンのシーケンスに平坦化
            # (SAM ViT-H: C=1280, H=W=64, シーケンス長 = 4096パッチ)
            vision_hidden_states = image_embeds.flatten(2).permute(0, 2, 1)  # [B, 4096, 1280]
            
            # カスタムビジョン埋め込みを保存
            self._last_image_embeds = image_embeds
        
        # ベースモデルでフォワードパスを実行
        # カスタムビジョン埋め込みがある場合は使用
        if hasattr(self, 'model') and self.model is not None:
            # モデル.generateの出力のhidden_statesを使用するため、return_dictを設定
            kwargs_to_pass = {**kwargs, "return_dict": True}
            if vision_hidden_states is not None:
                kwargs_to_pass['vision_hidden_states'] = vision_hidden_states
                
            # ベースモデルの前方伝播を呼び出す
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=None,  # 自前のSAM埋め込みを使うためNoneを渡す
                labels=labels,
                **kwargs_to_pass
            )
        else:
            # ベースモデルがない場合はエラー
            raise ValueError("Base model has not been initialized yet!")
        
        # 訓練中かつマスクリストがあれば、セグメンテーション損失を計算
        if not inference and masks_list is not None and len(masks_list) > 0 and self.seg_token_idx is not None and hasattr(self, 'visual_model'):
            # マスク生成のためのセグメントトークン埋め込みを取得
            batch_size = outputs.logits.shape[0]
            # モデル出力からhidden statesを取得（出力形式によって異なる）
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                # transformersの標準出力形式
                if isinstance(outputs.hidden_states, tuple):
                    hidden_states = outputs.hidden_states[-1]  # 最後の層の隠れ状態
                else:
                    hidden_states = outputs.hidden_states
            else:
                # 隠れ状態が直接利用できない場合
                hidden_states = None
                
            if hidden_states is None:
                # 損失計算をスキップ
                return outputs
            
            # セグメントトークンの位置を検索
            seg_token_pos = (input_ids == self.seg_token_idx).nonzero()
            if len(seg_token_pos) == 0:
                # セグメントトークンがない場合は損失計算をスキップ
                return outputs
                
            # 各バッチサンプルのセグメントトークン埋め込みを抽出
            seg_token_embeds = []
            for i in range(batch_size):
                batch_indices = seg_token_pos[:, 0] == i
                if not batch_indices.any():
                    continue  # このバッチサンプルにセグメントトークンがない
                    
                # このバッチサンプルのセグメントトークン位置
                pos = seg_token_pos[batch_indices, 1]
                # 各位置の埋め込みを取得（複数のセグメントトークンがある場合）
                for p in pos:
                    seg_token_embed = hidden_states[i, p]  # [hidden_dim]
                    # プロジェクションレイヤーを通過
                    for layer in self.text_hidden_fcs:
                        seg_token_embed = layer(seg_token_embed)  # [out_dim]
                    seg_token_embeds.append(seg_token_embed)
            
            # セグメントトークン埋め込みがない場合は損失計算をスキップ
            if len(seg_token_embeds) == 0:
                return outputs
                
            # 損失計算
            seg_token_embeds = torch.stack(seg_token_embeds)  # [num_seg_tokens, out_dim]
            
            # SAMのマスクデコーダを使用してマスクを生成
            total_ce_loss = 0
            total_dice_loss = 0
            total_masks = 0
            
            # 各マスクに対する損失を計算
            for i, (gt_mask, resize_info) in enumerate(zip(masks_list, resize_list)):
                if i >= len(seg_token_embeds):
                    break
                    
                # 埋め込みをマスクデコーダーに供給してマスクを生成
                mask_embed = seg_token_embeds[i].unsqueeze(0)  # [1, out_dim]
                
                # SAMのマスクデコーダを使用
                with torch.set_grad_enabled(self.train_mask_decoder):
                    # 同じバッチの元画像特徴を取得
                    batch_idx = seg_token_pos[i, 0]
                    img_embeddings = self._last_image_embeds[batch_idx:batch_idx+1]
                    
                    # SAMのプロンプトエンコーダとマスクデコーダを使用
                    sparse_embeddings = mask_embed.unsqueeze(0)  # [1, 1, out_dim]
                    dense_embeddings = self.visual_model.prompt_encoder.no_mask_embed.weight.reshape(1, 1, -1).expand(1, img_embeddings.shape[-2] * img_embeddings.shape[-1], -1)
                    
                    # マスクデコーダの出力
                    decoder_out = self.visual_model.mask_decoder(
                        image_embeddings=img_embeddings,
                        image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    
                    # マスクを元の画像サイズにリサイズ
                    pred_mask = decoder_out.masks.squeeze(1)  # [1, H, W]
                    
                    # 元のサイズにリサイズ
                    pred_mask = F.interpolate(
                        pred_mask.unsqueeze(1),  # [1, 1, H, W]
                        size=(resize_info[0], resize_info[1]),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)  # [1, H, W]
                    
                    # グラウンドトゥルースマスクの処理
                    gt_mask = gt_mask.to(pred_mask.device)
                    
                    # 損失計算
                    ce_loss = sigmoid_ce_loss(pred_mask, gt_mask, 1)
                    dice_loss_val = dice_loss(pred_mask, gt_mask, 1)
                    
                    total_ce_loss += ce_loss
                    total_dice_loss += dice_loss_val
                    total_masks += 1
            
            # 平均損失を計算
            if total_masks > 0:
                avg_ce_loss = total_ce_loss / total_masks
                avg_dice_loss = total_dice_loss / total_masks
                # 重み付けされた合計損失
                seg_loss = self.ce_loss_weight * avg_ce_loss + self.dice_loss_weight * avg_dice_loss
                # 言語モデル損失に追加
                if hasattr(outputs, 'loss') and isinstance(outputs.loss, torch.Tensor):
                    outputs.loss = outputs.loss + seg_loss
                else:
                    outputs.loss = seg_loss
        
        return outputs
        
    def generate(self, **kwargs):
        """
        モデル生成のラッパーメソッド
        """
        if hasattr(self, 'model') and self.model is not None:
            # output_hidden_statesを指定して、隠れ状態を取得
            if 'output_hidden_states' not in kwargs:
                kwargs['output_hidden_states'] = True
            
            # BFloat16データ型の問題を解決するためのパッチ
            orig_dtype = None
            if hasattr(self.model, 'dtype') and self.model.dtype == torch.bfloat16:
                print("BFloat16からFloat16へデータ型を変換して生成します")
                # モデルのデータ型を一時的にfloat16に変換
                orig_dtype = self.model.dtype
                # 代わりにfloat16またはfloat32を使用
                self.model = self.model.to(torch.float16)
            
            try:
                # 生成実行
                return self.model.generate(**kwargs)
            except RuntimeError as e:
                if "triu_tril_cuda_template" in str(e) and "BFloat16" in str(e):
                    print("BFloat16エラーが発生したため、float32で再試行します")
                    # BFloat16エラーの場合、float32でもう一度試す
                    self.model = self.model.to(torch.float32)
                    return self.model.generate(**kwargs)
                else:
                    # その他のエラーは再度発生させる
                    raise
            finally:
                # 元のデータ型に戻す
                if orig_dtype is not None:
                    self.model = self.model.to(orig_dtype)
        else:
            raise ValueError("Base model has not been initialized yet!")
        
    def generate_masks(
        self,
        image=None,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        **kwargs
    ):
        try:
            print("LISA.generate_masksが呼び出されました")
            
            # 実行時のデバイス情報を表示
            device = next(self.parameters()).device
            print(f"モデルデバイス: {device}")
            
            # SAMエンコーダがメモリにあるか確認
            if not hasattr(self, "visual_model") or self.visual_model is None:
                print("警告: SAM視覚モデルが初期化されていません。初期化を試みます。")
                self._init_sam_model()
            
            # SAMエンコーダが正しいデバイスにあるか確認
            sam_device = next(self.visual_model.parameters()).device if hasattr(self, "visual_model") else None
            print(f"SAMモデルデバイス: {sam_device}")
            
            # 必要に応じてSAMモデルをGPUに移動（メモリ効率のため）
            if sam_device != device and device.type == "cuda":
                print(f"SAMモデルを{sam_device}から{device}に移動します")
                try:
                    # GPUに移動を試みる
                    self.visual_model.to(device)
                    print("SAMモデルを正常にGPUに移動しました")
                except Exception as move_error:
                    print(f"SAMモデルの移動中にエラー: {move_error}")
                    print("CPUで処理を続行します")
            
            # パラメータの整理とデバイスの統一
            kwargs = self._prepare_inputs_for_generation(input_ids, attention_mask=attention_mask, **kwargs)
            
            # generate関数を呼び出して応答を生成
            outputs = super().generate(**kwargs)
            
            # セグメンテーションが必要かどうかを判断
            if "<SEG>" not in self.tokenizer.decode(input_ids[0]) and "<SEG>" not in self.tokenizer.decode(outputs[0]):
                print("セグメンテーショントークンがないため、通常の生成結果を返します")
                return outputs
            
            # 画像が提供されたかどうかチェック
            if image is None and "pixel_values" in kwargs:
                # pixel_valuesから画像情報を使用
                pixel_values = kwargs["pixel_values"]
                print(f"pixel_values形状: {pixel_values.shape if pixel_values is not None else None}")
            elif "images" in kwargs:
                # imagesキーがある場合はそれを使用
                print("imagesキーから画像を使用します")
                from PIL import Image
                import torch
                
                # PILイメージをテンソルに変換
                if isinstance(kwargs["images"], Image.Image):
                    # 画像の前処理
                    from torchvision import transforms
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                    ])
                    pixel_values = transform(kwargs["images"]).unsqueeze(0).to(device)
                    print(f"PILイメージからpixel_values形状: {pixel_values.shape}")
            elif image is not None:
                # 直接画像パラメータが渡された場合
                print("image引数から画像を使用します")
                from PIL import Image
                import torch
                
                # PILイメージをテンソルに変換
                if isinstance(image, Image.Image):
                    # 画像の前処理
                    from torchvision import transforms
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                    ])
                    pixel_values = transform(image).unsqueeze(0).to(device)
                    print(f"image引数からpixel_values形状: {pixel_values.shape}")
            
            # 画像がないなら早期リターン
            if pixel_values is None:
                print("画像が提供されていないため、通常の生成結果を返します")
                return outputs
            
            # 出力の中から<SEG>トークンのインデックスを検索
            tokenizer = self.tokenizer
            seg_token_id = tokenizer.convert_tokens_to_ids("<SEG>")
            seg_token_indices = []
            
            for i, sequence in enumerate(outputs):
                for j, token_id in enumerate(sequence):
                    if token_id == seg_token_id:
                        seg_token_indices.append(j)
            
            if not seg_token_indices:
                print("<SEG>トークンが出力に見つかりませんでした。通常の生成結果を返します")
                return outputs
            
            print(f"<SEG>トークンインデックス: {seg_token_indices}")
            
            # SAMモデルを使用した処理
            with torch.no_grad():
                # 画像エンコーディング
                try:
                    image_embeddings = self.visual_model.image_encoder(pixel_values)
                    print(f"SAM画像エンベディング形状: {image_embeddings.shape}")
                except Exception as encode_error:
                    print(f"画像エンコード中にエラー: {encode_error}")
                    import traceback
                    traceback.print_exc()
                    # エラーが発生した場合は元の出力を返す
                    return outputs
                
                # SAMデコードで生成されたテキストを使用してマスクを生成
                try:
                    # 元の画像サイズを取得
                    orig_size = kwargs.get("orig_size", None)
                    if orig_size is None and isinstance(image, Image.Image):
                        print("画像サイズを自動検出します")
                        width, height = image.size
                        orig_size = (height, width)
                    
                    if orig_size:
                        print(f"元の画像サイズ: {orig_size}")
                    
                    # マスクデコーディング
                    masks = self.get_masks(
                        image_embeddings=image_embeddings,
                        output_tokens=outputs,
                        seg_token_indices=seg_token_indices,
                        original_sizes=[orig_size] if orig_size else None,
                    )
                    
                    # マスク情報を表示
                    if masks is not None:
                        print(f"生成されたマスク: {len(masks)}個")
                    else:
                        print("マスク生成に失敗しました")
                        
                    # 生成結果を返す（マスクは別途保存/処理される）
                    return outputs
                
                except Exception as mask_error:
                    print(f"マスク生成中にエラー: {mask_error}")
                    import traceback
                    traceback.print_exc()
                    return outputs
                
        except Exception as e:
            print(f"generate_masks関数内でエラーが発生: {e}")
            import traceback
            traceback.print_exc()
            # エラーの場合は通常の生成結果を返す
            return super().generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def load_state_dict(self, state_dict, strict=False):
        """
        モデルの状態辞書をロードするメソッド
        
        Args:
            state_dict: ロードする状態辞書
            strict: 厳密なロードを行うかどうか
        """
        # プロパティとしてbase_modelが定義されていない場合は、通常のロード処理を行う
        if not hasattr(self, 'base_model'):
            return super().load_state_dict(state_dict, strict=strict)
        
        # 既に基本モデルが存在する場合は、そのままの状態を維持
        return self

    @property 
    def base_model(self):
        """基本モデルのプロパティ"""
        return self.model

    @base_model.setter
    def base_model(self, model):
        """基本モデルの設定"""
        self.model = model

    def _init_sam_model(self, sam_checkpoint=None, model_type="vit_h"):
        print("SAM視覚モデルを初期化中...")
        if model_type == "vit_h":
            # VIT_H（デフォルト）の場合はローカルチェックポイントを使用
            try:
                if sam_checkpoint is not None:
                    print(f"SAMチェックポイントをロード中: {sam_checkpoint}")
                    if os.path.exists(sam_checkpoint):
                        checkpoint_dict = torch.load(sam_checkpoint)
                        self.visual_model = build_sam_vit_h()
                        self.visual_model.load_state_dict(checkpoint_dict, strict=False)
                    else:
                        print(f"警告: SAMチェックポイントが見つかりません: {sam_checkpoint}")
                        print("デフォルトのSAMモデルを使用します")
                        # チェックポイントがない場合、HuggingFaceからダウンロード
                        from segment_anything import sam_model_registry
                        self.visual_model = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
                else:
                    # チェックポイントが指定されていない場合
                    print("デフォルトのSAMモデルを使用します")
                    # HuggingFaceからダウンロード
                    from segment_anything import sam_model_registry
                    self.visual_model = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
            except Exception as e:
                print(f"SAMモデルのロード中にエラー: {e}")
                traceback.print_exc()
                print("代替方法を試行中...")
                try:
                    # より一般的な初期化方法
                    from segment_anything import sam_model_registry
                    self.visual_model = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
                except Exception as e2:
                    print(f"代替方法でもエラー: {e2}")
                    traceback.print_exc()
                    # 最後の手段
                    print("SAMモデルの初期化に失敗したため、推論は画像セグメンテーションなしで実行されます")
                    self.visual_model = None
        else:
            # 代替モデルタイプ（例：vit_lなど）
            from segment_anything import sam_model_registry
            self.visual_model = sam_model_registry[model_type](checkpoint=f"sam_{model_type}.pth")
        
        # モデルを推論モードに設定
        if self.visual_model is not None:
            self.visual_model.eval()
            print("SAM視覚モデルが正常に初期化されました")
        
        return self.visual_model
    
    def get_masks(
        self,
        image_embeddings: torch.FloatTensor,
        output_tokens: torch.LongTensor,
        seg_token_indices: List[int],
        original_sizes: Optional[List[Tuple[int, int]]] = None
    ) -> List[torch.Tensor]:
        """
        セグメントトークンの埋め込みを使用してマスクを生成します。
        
        Args:
            image_embeddings: SAMエンコーダの出力画像埋め込み
            output_tokens: 生成されたテキストトークンID
            seg_token_indices: <SEG>トークンのインデックス
            original_sizes: 元の画像サイズ（高さ、幅）のリスト
            
        Returns:
            生成されたマスクのリスト
        """
        print(f"get_masks: 画像埋め込み形状={image_embeddings.shape}, トークン長={len(output_tokens[0])}")
        
        # SAMモデルが初期化されているか確認
        if not hasattr(self, 'visual_model') or self.visual_model is None:
            print("警告: SAMモデルが初期化されていません")
            return []
        
        # モデルのフォワードパスを実行して埋め込みを取得
        with torch.no_grad():
            # メモリ効率のためCPU→GPU転送を最適化
            device = image_embeddings.device
            
            # 生成出力の隠れ状態を取得
            hidden_states = None
            
            # 入力IDからセグメントトークンの隠れ状態を抽出
            seg_hidden_states = []
            
            # 各<SEG>トークンのインデックスからそれに対応する隠れ状態を取得
            for seg_token_idx in seg_token_indices:
                # 安全対策：インデックスが存在するか確認
                if not 0 <= seg_token_idx < output_tokens.shape[1]:
                    print(f"警告: セグメントトークンのインデックス {seg_token_idx} が範囲外です")
                    continue
                
                # 隠れ状態が未取得の場合、取得を試みる
                if hidden_states is None:
                    # 出力IDを使用して推論実行（隠れ状態を取得するため）
                    model_outputs = self.model(
                        input_ids=output_tokens,
                        output_hidden_states=True,
                        return_dict=True
                    )
                    hidden_states = model_outputs.hidden_states
                
                # 最後のレイヤーの隠れ状態を取得
                if isinstance(hidden_states, tuple):
                    last_hidden_states = hidden_states[-1]
                else:
                    last_hidden_states = hidden_states
                    
                # <SEG>トークンの埋め込みを取得
                seg_hidden_state = last_hidden_states[0, seg_token_idx]
                seg_hidden_states.append(seg_hidden_state)
            
            # <SEG>トークンの隠れ状態をプロジェクション
            sparse_embeddings = []
            dense_embeddings = []
            
            # mm_projectorがない場合、デフォルトの次元変換関数を定義
            if not hasattr(self, 'mm_projector') or self.mm_projector is None:
                print("警告: mm_projectorが見つかりません。デフォルトの変換を使用します。")
                # SAM埋め込み用の次元に変換するシンプルな線形変換
                def default_projection(hidden_state):
                    # 入力次元をSAMのpoint_embedのサイズ(256)に合わせる
                    out_dim = 256
                    # 元の次元（Llama隠れ状態）
                    in_dim = hidden_state.shape[-1]
                    # GPU上でランダムな線形変換を生成
                    if not hasattr(self, '_default_projector'):
                        # キャッシュして再利用（ランダムな初期化を維持）
                        self._default_projector = nn.Linear(in_dim, out_dim).to(device)
                    return self._default_projector(hidden_state)
                
                projector = default_projection
            else:
                projector = self.mm_projector
            
            for seg_hidden_state in seg_hidden_states:
                # 各セグメントトークンのプロジェクション実行
                try:
                    sparse_emb = projector(seg_hidden_state).unsqueeze(0)
                    sparse_embeddings.append(sparse_emb)
                    dense_embeddings.append(None)  # SAMはプロンプトポイントのみを使用
                except Exception as e:
                    print(f"プロジェクション中にエラー: {e}")
                    # エラー時は元の埋め込みをスキップ
                    continue
            
            # セグメント埋め込みが取得できない場合はエラー
            if len(sparse_embeddings) == 0:
                print("警告: セグメント埋め込みを取得できませんでした")
                return []
            
            # SAMデコーダを使用してマスクを生成
            masks = []
            
            # original_sizesがない場合、デフォルトサイズを使用
            if original_sizes is None or len(original_sizes) == 0:
                # 入力画像のサイズを取得（SAMデコーダのデフォルトサイズ）
                # 一般的なSAMデフォルトサイズは1024x1024
                default_size = (1024, 1024)
                original_sizes = [default_size for _ in range(len(sparse_embeddings))]
                print(f"元の画像サイズが指定されていないため、デフォルトサイズを使用: {default_size}")
            
            # 各セグメントプロンプトに対してマスクを生成
            for i, (sparse_emb, dense_emb) in enumerate(zip(sparse_embeddings, dense_embeddings)):
                try:
                    # インデックスが範囲外の場合はスキップ
                    if i >= len(original_sizes):
                        original_size = original_sizes[-1]  # 最後のサイズを使用
                    else:
                        original_size = original_sizes[i]
                    
                    # SAM Mask Decoderを実行
                    mask_output = self.visual_model.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=False,
                    )
                    
                    # マスクを取得
                    mask = mask_output["low_res_logits"]
                    
                    # マスクをシグモイド関数で確率に変換
                    mask = torch.sigmoid(mask)
                    
                    # 必要に応じてしきい値処理
                    binary_mask = (mask > 0.5).float()
                    
                    # マスクを元の解像度にリサイズ
                    # 元の画像サイズがある場合はリサイズする
                    if original_size and original_size != binary_mask.shape[-2:]:
                        try:
                            import torch.nn.functional as F
                            binary_mask = F.interpolate(
                                binary_mask, size=original_size, mode="bilinear", align_corners=False
                            )
                        except Exception as resize_error:
                            print(f"マスクのリサイズ中にエラー: {resize_error}")
                            # エラー時は元のサイズを維持
                    
                    masks.append(binary_mask.squeeze())
                    
                except Exception as mask_error:
                    print(f"マスク{i}の生成中にエラー: {mask_error}")
                    import traceback
                    traceback.print_exc()
            
            # 各マスクの形状を表示
            for i, mask in enumerate(masks):
                print(f"マスク {i} の形状: {mask.shape}")
            
            # メモリの解放（大規模なテンソル）
            del hidden_states
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # マスクの保存や視覚化はchat.pyで行われる
            return masks
