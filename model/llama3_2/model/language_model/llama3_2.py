from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig, 
    AutoProcessor, 
    AutoModelForVision2Seq, 
    MllamaForConditionalGeneration,
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationMixin
)
from transformers.modeling_outputs import CausalLMOutputWithPast

# ここで定義したLlama3VisionMetaModelとLlama3VisionMetaForCausalLMは
# LISA.pyで使用する統合モデルの基底クラスとなります


class Llama3VisionMetaModel(nn.Module):
    """
    Llama3.2 Vision Meta Model for LISA integration.
    This class serves as a wrapper around the AutoModelForVision2Seq model.
    """
    
    def __init__(self, config, model_name=None, **kwargs):
        super().__init__()
        # モデルの設定
        self.config = config
        # モデル名を設定
        self.model_name = model_name or "meta-llama/Llama-3.2-11B-Vision-Instruct"
        # Note: torch_dtypeは一時変数として使用し、インスタンス変数として保存しない
        self._torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)
        
        # プロセッサーとモデルの読み込み
        self.processor = None  # プロセッサーは別途初期化
        self.vision_tower = None  # ビジョンタワーもLISAモデルで別途扱う
    
    def get_processor(self):
        """
        プロセッサーを取得します。初期化されていない場合は初期化します。
        """
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
        return self.processor


class Llama3VisionForCausalLM(PreTrainedModel, GenerationMixin):
    """
    Llama3.2 Vision Model for causal language modeling with LISA integration.
    This class serves as a wrapper around MllamaForConditionalGeneration for LISA.
    """
    
    def __init__(self, config, model_id=None, **kwargs):
        # 設定を取得して親クラスを初期化
        super().__init__(config)
        
        # モデルIDを設定（configから取得するか、引数から取得）
        self.model_id = model_id or "meta-llama/Llama-3.2-11B-Vision-Instruct"
        
        # Note: 実際のモデルの初期化はLISAForCausalLMで行います
        # ここでは、スケルトンだけを提供します
        self.model = None
        self.processor = None
        
        # LISAモデルでの統合に必要な属性を初期化
        self.vision_tower = None  # LISAで別途扱う
        
    def get_model(self):
        """モデルを返します"""
        return self.model
    
    def get_processor(self):
        """
        プロセッサーを取得します。初期化されていない場合は初期化します。
        """
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
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
        if self.get_output_embeddings() is not None:
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
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Llama3.2 Visionモデルのforward関数です。
        LISAモデルで使われるインターフェースに合わせています。
        """
        # 画像があればプロセッサを使って処理
        if images is not None:
            processor = self.get_processor()
            # 適切な形式でプロンプトを作成（テキストは後で追加するためここでは空にする）
            texts = [""] * images.size(0)  # バッチサイズ分の空テキスト
            
            # 画像を処理
            processed_inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
            
            # 必要に応じてラベルを設定（ターゲットシーケンス）
            if labels is not None:
                processed_inputs["labels"] = labels
                
            # モデルにそのまま渡す
            outputs = self.model(**processed_inputs, **kwargs)
            
            return outputs
        else:
            # 画像がない場合はテキスト入力のみで実行
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs
            )
            
            return outputs
    
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
        生成時の入力準備関数です。
        この関数はテキスト生成時のパラメータ設定に使用されます。
        """
        # 標準的な入力準備
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            "images": images,
            **kwargs
        }
        
        return model_inputs 