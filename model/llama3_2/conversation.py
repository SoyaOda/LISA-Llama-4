import dataclasses
from enum import Enum, auto
from typing import List, Optional, Tuple

class SeparatorStyle(Enum):
    """会話フォーマットスタイルの列挙型"""
    LLAMA_2 = auto()
    PLAIN = auto()

@dataclasses.dataclass
class Conversation:
    """すべての会話履歴を保持し、モデル用にフォーマットできるクラス"""
    system: str
    roles: Tuple[str, str]  # (user_role, assistant_role)のタプル
    messages: List[Tuple[str, Optional[str]]]  # (role, message)ペアのリスト。messageはプレースホルダの場合はNone。
    offset: int = 0
    sep_style: SeparatorStyle = SeparatorStyle.LLAMA_2
    sep: str = "<s>"    # LLAMA_2スタイルの場合はBOSトークン文字列、PLAINの場合は改行など
    sep2: str = "</s>"  # LLAMA_2スタイルの場合はEOSトークン文字列、PLAINの場合はNoneか2番目の区切り文字
    version: str = "llama3_2"

    def get_prompt(self) -> str:
        """会話をモデルが処理できるプロンプト文字列にフォーマットする"""
        # システムメッセージから始める（存在する場合）適切にフォーマット
        if self.sep_style == SeparatorStyle.LLAMA_2:
            # LLAMA_2スタイルは特殊な[INST]タグとシステムプロンプトを<<SYS>>でラップする
            prompt = ""
            user_role, assistant_role = self.roles
            for i, (role, message) in enumerate(self.messages):
                # ロールラベルが期待値と一致することを確認（小文字で比較）
                if i == 0:
                    # 最初のメッセージはユーザーからのものであることを期待
                    assert role.lower() == user_role.lower(), "LLAMA_2スタイルでは最初のメッセージはユーザーからでなければなりません"
                    # システムプロンプト部分を準備
                    sys_msg = self.system.strip()
                    sys_section = f"<<SYS>>\n{sys_msg}\n<</SYS>>\n\n" if len(sys_msg) > 0 else ""
                    # 最初のユーザーメッセージが提供されている場合
                    user_content = message or ""
                    prompt += f"{self.sep}[INST] {sys_section}{user_content} [/INST]"
                    # ここで終了EOSを追加しない；アシスタントの応答が続く
                else:
                    if role.lower() == user_role.lower():
                        # 新しいユーザーメッセージは新しい[INST]ブロックを開始
                        user_content = message or ""
                        prompt += f"{self.sep}[INST] {user_content} [/INST]"
                        # [/INST]の後、アシスタントが応答することを期待
                    elif role.lower() == assistant_role.lower():
                        # アシスタントメッセージはEOSで[INST]ブロックを閉じる
                        assistant_content = message or ""
                        prompt += f" {assistant_content} {self.sep2}"
            # 最後のメッセージがユーザーからで、まだアシスタントの応答がない場合
            # （つまり、最後のエントリのロールがユーザーでメッセージが提供されており、
            # モデルがアシスタントの回答を生成することを望む場合）、モデルが続行できるようにプロンプトを開いたままにします。
            # LLAMA_2フォーマットでは、最後の[/INST]（ユーザーメッセージ）の後に、アシスタントの応答や終了EOSを追加しません。
            # 最後のメッセージのロールを確認：
            if len(self.messages) > 0 and self.messages[-1][0].lower() == self.roles[0].lower():
                # 最後のメッセージはユーザー；誤って追加された可能性のある終了EOSを削除
                if prompt.endswith(self.sep2):
                    prompt = prompt[: -len(self.sep2)]
            return prompt
        elif self.sep_style == SeparatorStyle.PLAIN:
            # プレーンスタイル：単にロール名と改行で会話を連結
            prompt_lines = []
            if self.system:
                prompt_lines.append(self.system)
            for role, message in self.messages:
                role_name = role  # 与えられたロール文字列をプレフィックスとして使用（空でない場合）
                if message is None:
                    # メッセージがNoneの場合、ロール名の後に何も追加しない（モデルの応答を待つ）
                    prompt_lines.append(f"{role_name}:")
                else:
                    # ロール名を含む通常のメッセージライン
                    prompt_lines.append(f"{role_name}: {message}")
            # 行を改行区切り文字で結合
            return "\n".join(prompt_lines)
        else:
            raise ValueError(f"サポートされていない区切りスタイル: {self.sep_style}")

    def append_message(self, role: str, message: Optional[str]):
        """新しいメッセージ（ロールと内容）を会話履歴に追加する"""
        self.messages.append((role, message))

    def copy(self):
        """会話の浅いコピーを作成（テンプレートを再利用時に保存するため）"""
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=self.messages.copy(),
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
        )

# デフォルトの会話テンプレート（Llama3.2 Vision用）
def get_default_conv_template():
    """Llama3.2 Vision用のデフォルト会話テンプレートを返す"""
    return Conversation(
        system="あなたは役立つアシスタントで、画像とテキストを理解できます。",
        roles=("user", "assistant"),
        messages=[],
        sep_style=SeparatorStyle.LLAMA_2,
        sep="<s>",
        sep2="</s>",
        version="llama3_2"
    )

# 会話テンプレート辞書
conv_templates = {
    "llama3_2": get_default_conv_template(),
    "plain": Conversation(
        system="",
        roles=("user", "assistant"),
        messages=[],
        sep_style=SeparatorStyle.PLAIN,
        sep="\n",
        sep2="\n",
        version="plain"
    )
} 