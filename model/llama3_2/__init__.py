from .constants import IGNORE_INDEX, IMAGE_TOKEN, SEG_TOKEN
from .conversation import Conversation, conv_templates, SeparatorStyle
from .mm_utils import tokenizer_image_token, process_images, load_image_from_base64, KeywordsStoppingCriteria

# 便宜のためのデフォルト会話テンプレート
default_system_prompt = "あなたは役立つアシスタントで、画像とテキストを理解できます。"
conv_llama3_2 = Conversation(
    system=default_system_prompt,
    roles=("user", "assistant"),
    messages=[],
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
    version="llama3_2"
)
