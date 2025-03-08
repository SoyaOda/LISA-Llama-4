from typing import List, Optional, Union, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel, AutoModelForVision2Seq, AutoConfig, AutoProcessor

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .segment_anything import build_sam_vit_h
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
        初期化メソッド
        
        Args:
            config: モデルの設定
            **kwargs: 追加の引数（vision_pretrained, train_mask_decoder, out_dim, seg_token_idx）
        """
        super().__init__()
        
        # 基本設定
        self.config = config
        
        # LISAのカスタム引数を取得
        self.train_mask_decoder = kwargs.get("train_mask_decoder", False)
        self.out_dim = kwargs.get("out_dim", 256)
        self.vision_ckpt = kwargs.get("vision_pretrained", None)
        self.ce_loss_weight = kwargs.get("ce_loss_weight", 1.0)
        self.dice_loss_weight = kwargs.get("dice_loss_weight", 0.5)
        self.seg_token_idx = kwargs.get("seg_token_idx", None)
        
        # ベースとなるビジョン言語モデル
        self.model = None
        
        # SAMの視覚エンコーダを初期化
        if self.vision_ckpt:
            self.initialize_sam_encoder()
    
    @classmethod
    def from_vision_model(cls, vision_model_id, seg_token_idx=None, vision_pretrained=None, train_mask_decoder=False, 
                         out_dim=256, tokenizer=None, torch_dtype=None, device_map=None, 
                         quantization_config=None, ignore_mismatched_sizes=False, load_in_8bit=False, load_in_4bit=False, **kwargs):
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
            **kwargs: その他の引数
        """
        print("Step 1: Loading base vision-language model...")
        # 最初にトークナイザーをロードし拡張する
        if tokenizer is None:
            tokenizer = AutoProcessor.from_pretrained(vision_model_id)
        
        # <SEG>トークンをトークナイザーに追加
        if "<SEG>" not in tokenizer.tokenizer.get_vocab():
            print("Adding <SEG> token to tokenizer vocabulary")
            if hasattr(tokenizer, 'tokenizer'):
                tokenizer.tokenizer.add_special_tokens({"additional_special_tokens": ["<SEG>"]})
            else:
                tokenizer.add_special_tokens({"additional_special_tokens": ["<SEG>"]})
        
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
        if device_map is None:
            # 自動的にGPUとCPU間でレイヤーを配置
            device_map = "auto"
        
        # トークナイザーを拡張した後にモデルをロード
        vision_model = AutoModelForVision2Seq.from_pretrained(
            vision_model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            quantization_config=quantization_config,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
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
        """SAMの視覚エンコーダを初期化"""
        # SAMの視覚エンコーダ（ViT-H）を初期化
        print(f"Initializing SAM encoder from {self.vision_ckpt}...")
        self.visual_model = build_sam_vit_h(self.vision_ckpt)
        
        # すべてのSAMパラメータを凍結（マスクデコーダを除く）
        for param in self.visual_model.parameters():
            param.requires_grad = False
            
        if self.train_mask_decoder:
            # マスクデコーダを訓練可能に設定（セグメンテーションマスクを学習する場合）
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # プロジェクションレイヤー：LLM隠れ状態をマスク埋め込み次元にプロジェクション
        # モデル設定からhidden_sizeを取得
        if hasattr(self.config, 'text_config') and hasattr(self.config.text_config, 'hidden_size'):
            hidden_size = self.config.text_config.hidden_size
        elif hasattr(self.config, 'hidden_size'):
            hidden_size = self.config.hidden_size
        else:
            # デフォルト値として4096を使用（一般的なLlamaモデルの隠れ層サイズ）
            hidden_size = 4096
            
        print(f"Using hidden size: {hidden_size} for projection layers")
        self.text_hidden_fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, self.out_dim),
                nn.Dropout(0.0)
            )
        ])
    
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
        テキスト生成を行う
        ベースモデルのgenerateメソッドを呼び出す
        """
        if hasattr(self, 'model') and self.model is not None:
            # output_hidden_statesを指定して、隠れ状態を取得
            if 'output_hidden_states' not in kwargs:
                kwargs['output_hidden_states'] = True
                
            return self.model.generate(**kwargs)
        else:
            raise ValueError("Base model has not been initialized yet!")
        
    def generate_masks(
        self,
        images: torch.FloatTensor,
        input_ids: torch.LongTensor,
        seg_token_indices: List[int],
        original_sizes: List[Tuple[int, int]],
        **kwargs
    ) -> List[torch.Tensor]:
        """
        セグメントトークンの埋め込みを使用してマスクを生成します。
        
        Args:
            images: SAMエンコーダ用の入力画像
            input_ids: 入力テキストトークンID
            seg_token_indices: <SEG>トークンのインデックス
            original_sizes: 元の画像サイズ（高さ、幅）
            
        Returns:
            生成されたマスクのリスト
        """
        if not hasattr(self, 'visual_model') or self.visual_model is None:
            raise ValueError("SAM visual model has not been initialized yet!")
            
        # モデルのフォワードパスを実行して埋め込みを取得
        with torch.no_grad():
            # 画像をSAMのViTでエンコード
            img_embeds = self.visual_model.image_encoder(images)
            
            # 生成出力の隠れ状態を取得
            # 多くの場合、生成出力は既に渡されており、隠れ状態も含まれているため、改めてforward()を呼ぶ必要はない
            # ただし、明示的にここでモデルに入力を渡すことも可能（以下のコメントアウト部分）
            # outputs = self(input_ids=input_ids, **kwargs)
            # # 隠れ状態を抽出
            # if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            #     hidden_states = outputs.hidden_states[-1]
            # else:
            #     # 適切な構造で隠れ状態を取得できない場合
            #     raise ValueError("Cannot access hidden states from model outputs")
            
            # モデル自体から埋め込みレイヤーを使用して埋め込みを取得
            if hasattr(self.model, 'get_input_embeddings'):
                # 入力埋め込みを取得
                embed_layer = self.model.get_input_embeddings()
                hidden_states = embed_layer(input_ids)
            else:
                # 埋め込みレイヤーが直接アクセスできない場合
                raise ValueError("Cannot access embedding layer of the model")
            
            # <SEG>トークンの埋め込みを取得
            seg_hidden_states = []
            batch_size = input_ids.shape[0]
            
            for i in range(batch_size):
                for seg_idx in seg_token_indices:
                    if seg_idx >= input_ids.shape[1]:
                        continue
                    if input_ids[i, seg_idx] == self.seg_token_idx:
                        # <SEG>トークンの隠れ状態を取得
                        seg_hidden = hidden_states[i, seg_idx]
                        # プロジェクションレイヤを適用
                        for layer in self.text_hidden_fcs:
                            seg_hidden = layer(seg_hidden)
                        seg_hidden_states.append((i, seg_hidden))
            
            # 各<SEG>トークンに対してマスクを生成
            masks = []
            for batch_idx, seg_embed in seg_hidden_states:
                # SAMのマスクデコーダに埋め込みを渡してマスクを生成
                mask_embed = seg_embed.unsqueeze(0)  # [1, out_dim]
                
                # 同じバッチの画像埋め込みを取得
                img_embedding = img_embeds[batch_idx:batch_idx+1]
                
                # SAMのプロンプトエンコーダとマスクデコーダを使用
                sparse_embeddings = mask_embed.unsqueeze(0)  # [1, 1, out_dim]
                dense_embeddings = self.visual_model.prompt_encoder.no_mask_embed.weight.reshape(1, 1, -1).expand(
                    1, img_embedding.shape[-2] * img_embedding.shape[-1], -1
                )
                
                # マスクデコーダの出力
                decoder_out = self.visual_model.mask_decoder(
                    image_embeddings=img_embedding,
                    image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                
                # 生成されたマスク
                mask = decoder_out.masks.squeeze(1)  # [1, H, W]
                
                # 元のサイズにリサイズ
                orig_h, orig_w = original_sizes[batch_idx]
                mask = F.interpolate(
                    mask.unsqueeze(1),  # [1, 1, H, W]
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()  # [H, W]
                
                # シグモイド関数を適用して0-1の範囲に正規化
                mask = torch.sigmoid(mask)
                masks.append(mask)
                
        return masks

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
