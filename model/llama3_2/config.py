from transformers import PretrainedConfig
from typing import Optional, Union, List

class Llama32SAMConfig(PretrainedConfig):
    """
    Llama3.2 Vision + SAM統合モデルの設定クラス
    """
    model_type = "llama32_sam"

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
        precision: str = "bf16",
        train_mask_decoder: bool = False,
        out_dim: int = 256,
        hidden_size: int = 4096,
        vision_tower: str = "openai/clip-vit-large-patch14",
        mm_vision_tower: Optional[str] = None,
        mm_use_im_start_end: bool = True,
        seg_token_idx: Optional[int] = None,
        vision_pretrained: Optional[str] = None,
        vocab_size: int = 32000,
        max_position_embeddings: int = 4096,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs
    ):
        """
        Args:
            model_id: Hugging Faceモデル ID
            precision: 精度 ("fp32", "fp16", "bf16")
            train_mask_decoder: マスクデコーダを訓練するかどうか
            out_dim: 出力次元
            hidden_size: 隠れ層の次元
            vision_tower: ビジョンタワーのモデル名
            mm_vision_tower: マルチモーダルビジョンタワーのモデル名
            mm_use_im_start_end: 画像開始/終了トークンを使用するかどうか
            seg_token_idx: セグメンテーショントークンのID
            vision_pretrained: SAMビジョンモデルの事前学習済み重みのパス
            vocab_size: 語彙サイズ
            max_position_embeddings: 最大位置埋め込み数
            pad_token_id: パディングトークンID
            bos_token_id: 開始トークンID
            eos_token_id: 終了トークンID
        """
        self.model_id = model_id
        self.precision = precision
        self.train_mask_decoder = train_mask_decoder
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.vision_tower = vision_tower
        self.mm_vision_tower = mm_vision_tower if mm_vision_tower is not None else vision_tower
        self.mm_use_im_start_end = mm_use_im_start_end
        self.seg_token_idx = seg_token_idx
        self.vision_pretrained = vision_pretrained
        
        # 親クラスのコンストラクタを呼び出す
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            **kwargs
        ) 