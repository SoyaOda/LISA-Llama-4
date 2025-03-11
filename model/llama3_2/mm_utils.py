import base64
from io import BytesIO

import torch
from PIL import Image
from transformers import StoppingCriteria

from .constants import IMAGE_TOKEN_INDEX


def load_image_from_base64(image):
    """
    Load an image from a base64 string.
    """
    return Image.open(BytesIO(base64.b64decode(image)))


def process_images(images, processor, return_processor_output=True):
    """
    Process images with the Llama3.2 vision processor.
    
    Args:
        images: PIL Image or list of PIL Images
        processor: Llama3.2 vision processor
        return_processor_output: Whether to return the full processor output or just the pixel values
        
    Returns:
        Processed images ready for model input
    """
    if not isinstance(images, list):
        images = [images]
        
    # Use the processor to prepare images
    processed = processor(images=images, return_tensors="pt")
    
    if return_processor_output:
        return processed
    else:
        return processed.pixel_values


def get_model_name_from_path(model_path):
    """
    Extract model name from the model path.
    """
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    """
    Stopping criteria based on keywords.
    Stops generation when a keyword is encountered.
    """
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [
            keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids
        ]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0]:] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(
            output_ids[:, -offset:], skip_special_tokens=True
        )[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False


def prepare_messages_for_llama3_vision(images, prompt):
    """
    Prepare messages for Llama3.2 vision model.
    
    Args:
        images: List of PIL images
        prompt: Text prompt
        
    Returns:
        List of message dictionaries in the Llama3.2 vision format
    """
    if not isinstance(images, list):
        images = [images]
        
    content = []
    
    # Add all images
    for img in images:
        content.append({"type": "image"})
    
    # Add the text prompt
    content.append({"type": "text", "text": prompt})
    
    messages = [{"role": "user", "content": content}]
    return messages 