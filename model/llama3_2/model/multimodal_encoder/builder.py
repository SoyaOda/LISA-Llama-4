from .vision_encoder import Llama3VisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    """
    Llama3.2 Visionは内部にビジョンエンコーダーを持っているため、
    ここではダミーのビジョンタワーを返します。
    実際の画像処理はLlama3VisionForCausalLMで行われます。
    """
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )
    
    # Llama3.2 visionの場合
    if "llama" in vision_tower.lower() and "vision" in vision_tower.lower():
        return Llama3VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    # fallback（通常は使用されない）
    raise ValueError(f"Unknown vision tower: {vision_tower}") 