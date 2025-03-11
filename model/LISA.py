from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, AutoProcessor, AutoModelForVision2Seq, AutoConfig, GenerationMixin

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
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
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
        # vision_towerをmodel_nameとして渡す
        model_name = kwargs.get("vision_tower") or config.vision_tower
        super(LisaModel, self).__init__(config, model_name=model_name, **kwargs)

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


class LISAForCausalLM(Llama3VisionForCausalLM):
    def __init__(
        self,
        model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        **kwargs,
    ):
        # モデルIDからconfigを生成
        config = kwargs.pop("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(model_id)
        
        # セグメンテーショントークンIDを保存
        self.seg_token_idx = kwargs.pop("seg_token_idx", None)
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_use_im_patch_token = kwargs.pop("use_mm_use_im_patch_token", False)
            config.tune_mm_mlp_adapter = kwargs.pop("tune_mm_mlp_adapter", False)
            config.freeze_backbone = kwargs.pop("freeze_backbone", True)
            config.vision_tower = kwargs.pop("vision_tower", None)
            config.mm_vision_select_layer = kwargs.pop("mm_vision_select_layer", -1)
            config.pretrain_mm_mlp_adapter = kwargs.pop("pretrain_mm_mlp_adapter", None)
            config.mm_vision_select_feature = kwargs.pop("mm_vision_select_feature", "patch")
            config.vision_select_layer = kwargs.pop("vision_select_layer", -1)
            config.image_size = kwargs.pop("image_size", 1024)
            config.train_mask_decoder = kwargs.pop("train_mask_decoder", True)
            config.out_dim = kwargs.pop("out_dim", 256)
            config.select_layer = kwargs.pop("select_layer", -1)
            config.vision_pretrained = kwargs.pop("vision_pretrained", None)
            config.device_map = kwargs.pop("device_map", None)
            config.query_len = kwargs.pop("query_len", 1)
        
        # 親クラスの初期化 - ここでPreTrainedModelを初期化
        super().__init__(config, model_id=model_id)
        
        # torch_dtype設定
        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)
        device_map = kwargs.get("device_map", "auto")
        
        # デバッグ情報
        print(f"Loading Llama3.2 Vision model: {model_id}")
        print(f"  - torch_dtype: {torch_dtype}")
        print(f"  - device_map: {device_map}")
        
        try:
            # Llama3.2 Vision用のモデルを直接設定
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id, 
                torch_dtype=torch_dtype,
                device_map=device_map
            )
            print(f"  - Model loaded successfully: {type(self.model)}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # モデルIDを保存
        self.model_id = model_id
        
        # プロセッサを初期化
        self.processor = None

        # LISAモデルの初期化
        self.lisa_model = LisaModel(config, **kwargs)
        
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
        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
        # Llama3.2 visionでは入力形式が異なるため、マスクを調整
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
            dim=1,
        )

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
                    attention_mask=attention_masks[start_i:end_i],
                    return_tensors="pt",
                    padding=True,
                )
                
                # デバイスを合わせる
                batch_inputs = {k: v.to(self.model.device) for k, v in batch_inputs.items()}
                
                # モデルを実行
                output_i = self.model(**batch_inputs, output_hidden_states=True)
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            # 出力を結合
            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
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
                attention_mask=attention_masks,
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
            output_hidden_states = output.hidden_states

        # 以下はオリジナルのLISAと同様の処理
        hidden_states = []

        assert len(self.lisa_model.text_hidden_fcs) == 1
        hidden_states.append(self.lisa_model.text_hidden_fcs[0](output_hidden_states[-1]))

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
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
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            # 以下はオリジナルのLISAと同様の処理
            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            # Llama3.2 vision用にマスクを調整
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
                    seg_token_mask,
                ],
                dim=1,
            )

            hidden_states = []

            assert len(self.lisa_model.text_hidden_fcs) == 1
            hidden_states.append(self.lisa_model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
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
