import dataclasses
from enum import Enum, auto
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import base64
from io import BytesIO


class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    TWO = auto()
    PLAIN = auto()
    LLAMA_2 = auto()
    LLAMA_3 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history for Llama3.2 vision."""

    system: str
    roles: List[str]
    messages: List[Dict[str, Any]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.LLAMA_3
    sep: str = "###"
    sep2: str = None
    version: str = "llama3.2"

    skip_next: bool = False

    def get_prompt(self):
        """
        Get the prompt for Llama3.2 vision model.
        Uses the chat template format suitable for Llama3.2 vision model.
        """
        if self.sep_style == SeparatorStyle.LLAMA_3:
            formatted_messages = []
            for message in self.messages:
                if message["role"] == "system" and self.system:
                    formatted_messages.append({"role": "system", "content": self.system})
                else:
                    formatted_messages.append(message)
            
            # This is just a placeholder. In actual use, we'll use the processor.apply_chat_template
            # that handles the formatting properly for Llama3.2 vision
            return formatted_messages
        else:
            # Fallback to simpler format for non-Llama3 models
            ret = self.system + self.sep if self.system else ""
            for message in self.messages:
                role = message["role"]
                content = message["content"]
                if isinstance(content, list):
                    # For multimodal content
                    text_parts = []
                    for part in content:
                        if part["type"] == "text":
                            text_parts.append(part["text"])
                    content = " ".join(text_parts)
                ret += role + ": " + content + self.sep
            return ret

    def append_message(self, role, content):
        """
        Append a message to the conversation.
        
        Args:
            role: The role of the message sender (user, assistant, system)
            content: The content of the message, can be text or a list of content parts
        """
        message = {"role": role, "content": content}
        self.messages.append(message)

    def get_images(self):
        """
        Extract images from the conversation.
        
        Returns:
            List of image objects
        """
        images = []
        for message in self.messages:
            if message["role"] == "user":
                content = message["content"]
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "image":
                            # Handle image data - this is a simplified version
                            if "image_url" in item:
                                # In a real implementation, we would load the image from the URL
                                pass
                            elif "image" in item:
                                # Direct image object, already loaded
                                images.append(item["image"])
        return images

    def to_gradio_chatbot(self):
        """
        Convert the conversation to a format that can be displayed in a Gradio chatbot.
        
        Returns:
            List of (user, assistant) message tuples
        """
        ret = []
        for message in self.messages:
            role = message["role"]
            content = message["content"]
            
            if isinstance(content, list):
                # For multimodal content, extract text and images
                text_parts = []
                image_parts = []
                for part in content:
                    if part["type"] == "text":
                        text_parts.append(part["text"])
                    elif part["type"] == "image":
                        # In a real implementation, we would convert the image to a format suitable for Gradio
                        image_parts.append("Image placeholder")
                
                content = "\n".join(text_parts)
                if image_parts:
                    content = content + "\n" + "\n".join(image_parts)
            
            if role == "user":
                ret.append([content, None])
            elif role == "assistant" and ret:
                ret[-1][1] = content
            elif role == "assistant" and not ret:
                ret.append([None, content])
        return ret

    def copy(self):
        """
        Create a deep copy of the conversation.
        
        Returns:
            A new Conversation object with the same content
        """
        return Conversation(
            system=self.system,
            roles=self.roles.copy(),
            messages=[message.copy() for message in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
            skip_next=self.skip_next,
        )


# 会話テンプレート
conv_llama_3 = Conversation(
    system="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
           "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
           "Please ensure that your responses are socially unbiased and positive in nature. "
           "If a question is not clear or factually coherent, explain why instead of answering something not correct. "
           "If you don't know the answer to a question, please don't share false information.",
    roles=["user", "assistant"],
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.LLAMA_3,
    sep="",
)

# Llama3.2 Visionモデル用のデフォルトの会話テンプレート
conv_llama3_vision = Conversation(
    system="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
           "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
           "Please ensure that your responses are socially unbiased and positive in nature. "
           "If a question is not clear or factually coherent, explain why instead of answering something not correct. "
           "If you don't know the answer to a question, please don't share false information.",
    roles=["user", "assistant"],
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.LLAMA_3,
    sep="",
    version="llama3.2-vision",
)

# その他の互換性のための会話テンプレート
conv_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=["Human", "Assistant"],
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_llama_2 = Conversation(
    system="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
           "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
           "Please ensure that your responses are socially unbiased and positive in nature. "
           "If a question is not clear or factually coherent, explain why instead of answering something not correct. "
           "If you don't know the answer to a question, please don't share false information.",
    roles=["USER", "ASSISTANT"],
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

# 会話テンプレートの登録
conv_templates = {
    "llava_v1": conv_v1,
    "llava_llama_2": conv_llama_2,
    "llama_3": conv_llama_3,
    "llama3_vision": conv_llama3_vision,
}

# デフォルトの会話テンプレート
default_conversation = conv_templates["llama3_vision"] 