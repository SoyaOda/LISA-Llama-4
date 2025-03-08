from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MllamaForConditionalGeneration, AutoProcessor, PreTrainedModel

from .config import Llama32SAMConfig
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                     DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from ..segment_anything import build_sam_vit_h
from ..segment_anything.modeling import Sam


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    DICE損失の計算
    Args:
        inputs: マスク予測値
        targets: 正解マスク
        num_masks: マスクの数
        scale: スケーリング係数
        eps: 数値安定化のための微小値
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
    シグモイドクロスエントロピー損失の計算
    Args:
        inputs: マスク予測値
        targets: 正解マスク
        num_masks: マスクの数
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class Llama32SAMForCausalLM(PreTrainedModel):
    """
    Llama3.2 Vision + SAM統合モデルのメインクラス
    """
    config_class = Llama32SAMConfig
    
    def __init__(
        self,
        config: Llama32SAMConfig,
        **kwargs,
    ):
        # 設定からセグメンテーショントークンIDを取得
        if "seg_token_idx" in kwargs:
            self.seg_token_idx = kwargs.pop("seg_token_idx")
            config.seg_token_idx = self.seg_token_idx
        elif hasattr(config, "seg_token_idx") and config.seg_token_idx is not None:
            self.seg_token_idx = config.seg_token_idx
        else:
            raise ValueError("seg_token_idxを指定する必要があります")
            
        # 損失重みの設定
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        
        # 親クラスの初期化
        super().__init__(config)
        
        # Llama3.2 Visionモデルの初期化
        self.llama_model = MllamaForConditionalGeneration.from_pretrained(
            config.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16 if config.precision == "bf16" else 
                       torch.float16 if config.precision == "fp16" else torch.float32
        )
        
        # SAMモデルの初期化
        self.vision_pretrained = kwargs.get("vision_pretrained", None) or config.vision_pretrained
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True
        
        # プロジェクション層の定義
        in_dim = config.hidden_size
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
        
        # プロセッサの初期化
        self.processor = AutoProcessor.from_pretrained(config.model_id)
    
    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        """SAMのビジュアルエンコーダで特徴抽出"""
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        """転送関数"""
        if "past_key_values" in kwargs:
            return self.llama_model.forward(**kwargs)
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
        """モデルの転送関数（訓練時）"""
        # SAMのビジュアルエンコーダで画像特徴抽出
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        # SEGトークンのマスク作成
        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
        # IMAGE_TOKEN_INDEXのハック（先頭に1つの画像があると仮定）
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
            dim=1,
        )

        if inference:
            # 推論時の処理
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            
            # 画像をプロセッサで処理
            inputs = self.processor(
                images=images_clip,
                text="<|begin_of_text|><|image|>",
                return_tensors="pt"
            ).to(input_ids.device)
            
            # 入力IDsをLlama3.2のトークン形式に変換
            converted_input_ids = self.convert_input_ids_to_llama32(input_ids)
            
            # 出力を生成
            outputs = self.llama_model.generate(
                **inputs,
                input_ids=converted_input_ids,
                attention_mask=attention_masks,
                max_new_tokens=256,
                output_hidden_states=True,
                return_dict_in_generate=True
            )
            
            # 隠れ状態を取得
            output_hidden_states = outputs.hidden_states[-1]
            
            # SEGトークンの隠れ状態を特徴ベクトルに変換
            hidden_states = []
            assert len(self.text_hidden_fcs) == 1
            hidden_states.append(self.text_hidden_fcs[0](output_hidden_states))
            
        else:
            # 訓練時の処理
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
            
            # 画像をプロセッサで処理
            inputs = self.processor(
                images=images_clip,
                text=["<|begin_of_text|><|image|>"] * images_clip.shape[0],
                return_tensors="pt"
            ).to(input_ids.device)
            
            # 入力IDsをLlama3.2のトークン形式に変換
            converted_input_ids = self.convert_input_ids_to_llama32(input_ids)
            
            # Llama3.2モデルによる順伝播
            outputs = self.llama_model(
                **inputs,
                input_ids=converted_input_ids,
                attention_mask=attention_masks,
                labels=labels,
                output_hidden_states=True,
            )
            
            # 隠れ状態を取得
            output_hidden_states = outputs.hidden_states
            
            # SEGトークンの隠れ状態を特徴ベクトルに変換
            hidden_states = []
            assert len(self.text_hidden_fcs) == 1
            hidden_states.append(self.text_hidden_fcs[0](output_hidden_states[-1]))

        # 隠れ状態の合計
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]

        # SEGトークンのカウントとオフセット計算
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )

        # オフセットごとに予測埋め込みを分割
        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        # 画像埋め込みを取得
        image_embeddings = self.get_visual_embs(images)

        # マスク生成
        multimask_output = False
        pred_masks = []
        for i in range(len(pred_embeddings)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )

            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, iou_predictions = self.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            
            # リサイズリストが与えられていれば、それを使用
            if resize_list is not None:
                input_size = resize_list[i]
                # オリジナルサイズリストが与えられていれば、それを使用
                original_size = original_size_list[i] if 'original_size_list' in locals() else input_size
                pred_mask = self.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=input_size,
                    original_size=original_size,
                )
                pred_masks.append(pred_mask[:, 0])
            else:
                pred_masks.append(low_res_masks)

        # 訓練時のみ損失を計算
        if inference:
            return outputs, pred_masks
        else:
            # マスクに関する損失計算
            num_masks = label_list[0].shape[0] if len(label_list) > 0 else 0
            mask_bce_loss = 0
            mask_dice_loss = 0
            if num_masks > 0:
                for i, (pred_mask, gt_mask) in enumerate(zip(pred_masks, label_list)):
                    mask_bce_loss += sigmoid_ce_loss(pred_mask, gt_mask, num_masks)
                    mask_dice_loss += dice_loss(pred_mask, gt_mask, num_masks)
            
            # 損失の重み付け合計
            total_loss = outputs.loss  # 言語モデルの損失
            if num_masks > 0 and self.ce_loss_weight is not None:
                total_loss += self.ce_loss_weight * mask_bce_loss
            if num_masks > 0 and self.dice_loss_weight is not None:
                total_loss += self.dice_loss_weight * mask_dice_loss
            
            return {
                "loss": total_loss,
                "lm_loss": outputs.loss,
                "mask_bce_loss": mask_bce_loss if num_masks > 0 else None,
                "mask_dice_loss": mask_dice_loss if num_masks > 0 else None,
                "pred_masks": pred_masks,
            }

    def convert_input_ids_to_llama32(self, input_ids):
        """LLaVAの入力IDsをLlama3.2のフォーマットに変換"""
        # ここでLLaVAのトークン形式からLlama3.2のトークン形式に変換する処理を実装
        # これは実際のモデル実装に合わせて適宜調整が必要
        return input_ids

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
        """評価関数（chat.pyから呼び出される）"""
        with torch.no_grad():
            # 入力形式を調整
            device = input_ids.device
            
            # 画像入力の準備
            if isinstance(images_clip, dict):
                # 既にプロセッサで処理済みの場合
                inputs = images_clip  # そのまま使用
            else:
                # プロセッサで画像を処理
                inputs = self.processor(
                    images=images_clip,
                    text="<|begin_of_text|><|image|>",
                    return_tensors="pt"
                ).to(device)
            
            # input_idsが提供されていて、かつinputsに含まれていない場合のみ追加
            if "input_ids" not in inputs or (inputs["input_ids"] is not input_ids):
                inputs["input_ids"] = input_ids
            
            # 出力を生成
            outputs = self.llama_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            
            # 出力シーケンスとhidden_statesの取得
            # Llama3.2 Visionモデルは少し異なる出力形式を持っています
            output_ids = outputs.sequences
            
            # hidden_statesを取得
            # generate関数の出力形式：各デコードステップのhidden_statesが含まれる
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                all_hidden_states = outputs.hidden_states
                if isinstance(all_hidden_states, tuple):
                    # 最後の生成ステップの最終レイヤーのhidden_states
                    output_hidden_states = all_hidden_states[-1][-1]
                else:
                    # 別の形式の場合
                    output_hidden_states = all_hidden_states[-1]
            else:
                # hidden_statesが取得できない場合のフォールバック
                raise ValueError("モデルの出力にhidden_statesが含まれていません。output_hidden_states=Trueを設定してください。")

            # SEGトークンのマスク作成
            seg_token_mask = torch.zeros_like(output_ids[0], dtype=torch.bool)
            
            # 生成された出力シーケンスでSEGトークンを探す
            seg_positions = []
            for i in range(output_ids.shape[1]):
                if i < output_ids.shape[1] - 1:  # 配列の範囲外を防ぐ
                    if output_ids[0, i] == self.seg_token_idx:
                        seg_positions.append(i)
                        # マスクにSEGトークンの位置を記録
                        seg_token_mask[i] = True
            
            # SEGトークンが見つからない場合
            if not seg_positions:
                print("警告: 出力に[SEG]トークンが見つかりませんでした。")
                return output_ids, []

            # 隠れ状態の処理
            hidden_states = []
            assert len(self.text_hidden_fcs) == 1
            
            # 変換するhidden_statesを指定
            hidden_states.append(self.text_hidden_fcs[0](output_hidden_states))

            # 隠れ状態の合計
            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            
            # SEGトークンの埋め込みを取得
            pred_embeddings = []
            for seg_pos in seg_positions:
                if seg_pos < last_hidden_state.shape[1]:  # 配列の範囲外を防ぐ
                    pred_embeddings.append(last_hidden_state[0, seg_pos].unsqueeze(0))
                    
            if not pred_embeddings:
                print("警告: 有効なSEGトークンの埋め込みが取得できませんでした。")
                return output_ids, []

            # 画像埋め込みを取得
            image_embeddings = self.get_visual_embs(images)

            # マスク生成
            multimask_output = False
            pred_masks = []
            for i, pred_embedding in enumerate(pred_embeddings):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embedding.unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embedding.dtype)
                low_res_masks, iou_predictions = self.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[0].unsqueeze(0),
                    image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                
                # リサイズリストとオリジナルサイズリストの対応を確認
                resize_size = resize_list[0] if i >= len(resize_list) else resize_list[i]
                original_size = original_size_list[0] if i >= len(original_size_list) else original_size_list[i]
                
                pred_mask = self.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_size,
                    original_size=original_size,
                )
                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks

    def generate(
        self,
        images=None,
        input_ids=None,
        attention_mask=None,
        **kwargs
    ):
        """Llama3.2モデルを使用してテキスト生成"""
        device = input_ids.device if input_ids is not None else "cuda"
        
        # 画像入力の準備
        if images is not None:
            # 画像が辞書型の場合（すでにプロセッサで処理済み）
            if isinstance(images, dict):
                inputs = {k: v.to(device) for k, v in images.items()}
            else:
                # 画像をプロセッサで処理
                if input_ids is not None:
                    # 入力IDsがある場合はそれに応じたテキスト入力を構築
                    if hasattr(self.processor, 'tokenizer'):
                        input_text = "<|begin_of_text|><|image|>" + self.processor.tokenizer.decode(input_ids[0][input_ids[0] > 0])
                    else:
                        input_text = "<|begin_of_text|><|image|>"
                else:
                    # 入力IDsがない場合はデフォルトのテキスト入力
                    input_text = "<|begin_of_text|><|image|>"
                
                # プロセッサで画像とテキストを処理
                inputs = self.processor(
                    images=images,
                    text=input_text,
                    return_tensors="pt"
                ).to(device)
            
            # 入力IDsが提供されている場合は上書き
            if input_ids is not None:
                inputs["input_ids"] = input_ids
            
            # 生成
            outputs = self.llama_model.generate(
                **inputs,
                **kwargs
            )
            
            return outputs
        else:
            # 画像がない場合は通常の生成
            if input_ids is None:
                raise ValueError("画像または入力IDsが必要です")
                
            return self.llama_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            ) 