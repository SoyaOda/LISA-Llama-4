import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoConfig


class Llama3VisionTower(nn.Module):
    """
    Llama3.2 Vision用のビジョンタワークラス。
    Llama3.2 Visionは内部にビジョンエンコーダーを持っているため、
    このクラスは主にLISAモデルとのインターフェースを提供します。
    """
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        
        # Llama3.2 Visionのデフォルトモデル名
        if vision_tower is None or vision_tower == "":
            self.vision_tower_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        
        # 必要な設定を取得
        self.select_layer = getattr(args, "mm_vision_select_layer", -1)
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        
        # モデル設定の読み込み
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name)
    
    def load_model(self):
        """
        プロセッサのみを読み込みます。
        実際のモデルはLlama3VisionForCausalLMで管理されます。
        """
        self.image_processor = AutoProcessor.from_pretrained(self.vision_tower_name)
        # ビジョンタワーはLlama3.2 Vision内部で管理されるのでここではNoneに設定
        self.vision_tower = None
        self.is_loaded = True
    
    @torch.no_grad()
    def forward(self, images):
        """
        Llama3.2 Visionの場合、画像の特徴抽出はモデル内部で行われるため、
        ここでは画像の前処理のみを行います。
        
        Args:
            images: 入力画像（PIL画像またはテンソル）
            
        Returns:
            processed_images: 前処理済み画像
        """
        if not self.is_loaded:
            self.load_model()
        
        # 画像の前処理
        processed_images = self.image_processor(images=images, return_tensors="pt")
        return processed_images
    
    @property
    def dummy_feature(self):
        """
        ダミーの特徴量を返します。
        実際の特徴抽出はLlama3.2 Vision内部で行われます。
        """
        return torch.zeros(1, 1, self.hidden_size)
    
    @property
    def dtype(self):
        """
        モデルのデータ型を返します。
        """
        return torch.float16
    
    @property
    def device(self):
        """
        モデルのデバイスを返します。
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @property
    def config(self):
        """
        モデルの設定を返します。
        """
        if self.is_loaded:
            return self.cfg_only
        else:
            return AutoConfig.from_pretrained(self.vision_tower_name)
    
    @property
    def hidden_size(self):
        """
        隠れ層のサイズを返します。
        Llama3.2 vision 11B instructの場合は4096
        """
        return 4096
    
    @property
    def num_patches(self):
        """
        画像パッチの数を返します。
        Llama3.2 visionの内部実装に依存しますが、
        SAMとの統合のために必要なので適切な値を返します。
        """
        return 16 * 16  # 仮の値（256パッチ） 