from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from transformers import AutoProcessor

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                       DEFAULT_IMAGE_PATCH_TOKEN, IGNORE_INDEX,
                       IMAGE_TOKEN_INDEX)

from .multimodal_encoder.builder import build_vision_tower


class Llama3VisionArchModel:
    """
    Llama3.2 Vision用のベースアーキテクチャモデル。
    このクラスはLISAモデルで使用するLlama3.2 Visionの基本機能を提供します。
    """
    def __init__(self, config):
        super(Llama3VisionArchModel, self).__init__(config)
        
        # ビジョンタワーとプロジェクタの設定
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            # Llama3.2 visionではプロジェクターは内部で処理されるため、
            # このプロジェクターは主にSAMとの統合時に使用されます
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
    
    def get_vision_tower(self):
        """
        ビジョンタワーを取得します。
        """
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def initialize_vision_modules(self, model_args, fsdp=None):
        """
        ビジョンモジュールを初期化します。
        
        Args:
            model_args: モデル引数
            fsdp: FSDPの設定（使用する場合）
        """
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        
        # 設定を保存
        self.config.mm_vision_tower = vision_tower
        
        if hasattr(model_args, "mm_vision_select_layer"):
            self.config.mm_vision_select_layer = mm_vision_select_layer
        
        if hasattr(model_args, "mm_vision_select_feature"):
            self.config.mm_vision_select_feature = mm_vision_select_feature
        
        # LISAモデルでのSAMとの統合のために必要な設定
        pretrain_mm_mlp_adapter = getattr(model_args, "pretrain_mm_mlp_adapter", None)
        
        # ビジョンタワーを構築
        vision_tower = build_vision_tower(self.config)
        
        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower
        
        # Llama3.2 Visionでは画像プロジェクターは内部で処理されるため、
        # このプロジェクターはSAMとの統合時に使用されます
        hidden_size = self.config.hidden_size
        mm_hidden_size = getattr(self.config, "mm_hidden_size", vision_tower.hidden_size)
        
        if pretrain_mm_mlp_adapter is not None:
            mm_projector = nn.Linear(mm_hidden_size, hidden_size)
            self.mm_projector = mm_projector


class Llama3VisionArchForCausalLM(ABC):
    """
    Llama3.2 Vision用の因果言語モデルアーキテクチャ。
    このクラスはLISAモデルで使用するLlama3.2 Visionの拡張機能を提供します。
    """
    
    @abstractmethod
    def get_model(self):
        """
        モデルを取得します。
        派生クラスで実装される必要があります。
        """
        pass
    
    def get_vision_tower(self):
        """
        ビジョンタワーを取得します。
        """
        return self.get_model().get_vision_tower()
    
    def get_processor(self):
        """
        Llama3.2 Vision用のプロセッサーを取得します。
        """
        model_id = getattr(self, "model_id", "meta-llama/Llama-3.2-11B-Vision-Instruct")
        return AutoProcessor.from_pretrained(model_id)
    
    def encode_images(self, images):
        """
        画像をエンコードします。
        
        Args:
            images: 入力画像（PIL画像またはテンソル）
            
        Returns:
            encoded_images: エンコードされた画像（テンソル）
        """
        vision_tower = self.get_vision_tower()
        processor = self.get_processor()
        
        # Llama3.2 Visionでは画像処理はプロセッサーで行われます
        # この関数はLISAモデルとの互換性のために残しています
        with torch.no_grad():
            if type(images) is list:
                # 複数の画像がある場合
                image_features = []
                for image in images:
                    processed_image = processor(images=image, return_tensors="pt")
                    image_features.append(processed_image)
                return image_features
            else:
                # 単一の画像の場合
                processed_image = processor(images=images, return_tensors="pt")
                return processed_image
    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images
    ):
        """
        マルチモーダル入力の準備をします。
        この関数はLISAモデルとの互換性のために実装されていますが、
        Llama3.2 Visionでは内部処理が異なります。
        
        Args:
            input_ids: 入力ID
            attention_mask: アテンションマスク
            past_key_values: 過去のキー値
            labels: ラベル
            images: 入力画像
            
        Returns:
            dict: 処理された入力
        """
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[0] == 0:
            return input_ids, attention_mask, past_key_values, labels
        
        # Llama3.2 Visionでは画像と言語の統合は内部で処理されます
        # ここでは主にSAMとの統合のための処理を行います
        processor = self.get_processor()
        
        if type(images) is torch.Tensor:
            # 既にテンソル形式の場合
            image_features = images
        else:
            # 画像を処理
            with torch.no_grad():
                processed_images = self.encode_images(images)
                image_features = processed_images
        
        # 入力IDとアテンションマスクの形状をチェック
        # Llama3.2 Vision用に入力を調整する必要がある場合はここで行います
        
        # LISAモデルでの使用のため、画像特徴量も返します
        return input_ids, attention_mask, past_key_values, labels, image_features
    
    def initialize_vision_tokenizer(self, model_args, num_new_tokens):
        """
        ビジョントークナイザーを初期化します。
        Llama3.2 Visionでは通常不要ですが、LISAモデルとの互換性のために残しています。
        """
        pass  # Llama3.2 Visionではこの処理は不要 