"""
Llama 3.2 Vision用の会話フォーマットおよびプロンプトハンドラ
オリジナルのLLAVAコードを参考に構築
"""

import dataclasses
from enum import auto, Enum
from typing import List, Tuple


class SeparatorStyle(Enum):
    """会話の区切りスタイル"""
    LLAMA3 = auto()


@dataclasses.dataclass
class Conversation:
    """Llama 3.2 Vision用の会話クラス"""
    system: str
    roles: Tuple[str, str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.LLAMA3
    
    # 区切り文字とテンプレート
    sep = "\n"
    sep2 = " "
    
    def get_prompt(self):
        """会話をプロンプト文字列に変換"""
        if self.sep_style == SeparatorStyle.LLAMA3:
            ret = self.system + self.sep
            for i, (role, message) in enumerate(self.messages):
                if i == 0:
                    ret += message + self.sep
                    continue
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid separator style: {self.sep_style}")
    
    def append_message(self, role, message):
        """会話に新しいメッセージを追加"""
        self.messages.append([role, message])
    
    def to_gradio_chatbot(self):
        """Gradio用のチャットボットフォーマットに変換"""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if role == self.roles[0]:
                ret.append([msg, None])
            elif role == self.roles[1]:
                if len(ret) == 0:
                    ret.append([None, msg])
                else:
                    ret[-1][1] = msg
            else:
                raise ValueError(f"Invalid role: {role}")
        return ret
    
    def copy(self):
        """会話オブジェクトのコピーを作成"""
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
        )


# デフォルトシステムプロンプト
DEFAULT_SYSTEM_PROMPT = """あなたは役立つアシスタントです。"""


# Llama 3.2 Vision用の会話テンプレート
def get_default_conv_template():
    """
    Llama 3.2 Vision用のデフォルト会話テンプレートを取得
    
    <|image|>トークンは画像の位置を示します
    """
    system = DEFAULT_SYSTEM_PROMPT
    roles = ("User", "Assistant")
    messages = []
    offset = 0
    return Conversation(
        system=system,
        roles=roles,
        messages=messages,
        offset=offset,
        sep_style=SeparatorStyle.LLAMA3,
    )


# 会話テンプレートを辞書として定義
llama_3_template = get_default_conv_template()

# テンプレート辞書
conv_templates = {
    "llama_3": llama_3_template,
}

# デフォルトの会話テンプレート
default_conversation = llama_3_template 