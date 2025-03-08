"""
Llama3.2 Visionモデル用の定数
"""

# 損失計算で無視する位置のインデックス（パディングなど）
IGNORE_INDEX = -100  

# 画像プレースホルダトークン（Llama3.2 Visionでの画像表現用特殊トークン）
IMAGE_TOKEN = "<|image|>"

# 特殊トークン
SEG_TOKEN = "<SEG>"  # セグメンテーション結果を表すトークン 