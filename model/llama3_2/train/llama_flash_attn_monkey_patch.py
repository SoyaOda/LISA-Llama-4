import logging
from typing import List, Optional, Tuple

import torch
import transformers
from einops import rearrange
from torch import nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

try:
    from flash_attn.flash_attn_interface import \
        flash_attn_unpadded_qkvpacked_func
except ImportError:
    from flash_attn.flash_attn_interface import (
        flash_attn_varlen_qkvpacked_func as flash_attn_unpadded_qkvpacked_func,
    )

from flash_attn.bert_padding import pad_input, unpad_input


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    # [bsz, q_len, nh, hd]

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # Flash attention codes from
    # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

    # transform the data into the format required by flash attention
    qkv = torch.stack(
        [query_states, key_states, value_states], dim=2
    )  # [bsz, nh, 3, q_len, hd]
    qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]
    # We have disabled _prepare_decoder_attention_mask in LlamaModel forward (it's now a no-op)
    # We'll handle the attention bias ourself
    if attention_mask is not None:
        # attention_mask is q_len x k_len
        attention_mask = attention_mask[:, -q_len:]
        attention_mask = (
            ~attention_mask) if attention_mask.dtype == torch.bool else (
            1.0 - attention_mask)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [bsz, 1, 1, q_len, k_len]
        attention_mask = attention_mask.expand(-1, -1, q_len, -1)
        attention_mask = attention_mask.reshape(bsz * 1, q_len, -1)

        qkv = rearrange(qkv, "b s three h d -> (b h) s three d")
        max_seq_len = key_states.shape[-2]
        cu_q_lens = torch.arange(
            0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device
        )
        output = flash_attn_unpadded_qkvpacked_func(
            qkv, cu_q_lens, max_seq_len, 0.0, softmax_scale=None, causal=True
        )
        output = rearrange(output, "(b h) s d -> b s h d", h=self.num_heads)
    else:
        qkv = rearrange(qkv, "b s three h d -> (b h) s three d")
        max_seq_len = key_states.shape[-2]
        cu_q_lens = torch.arange(
            0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device
        )
        output = flash_attn_unpadded_qkvpacked_func(
            qkv, cu_q_lens, max_seq_len, 0.0, softmax_scale=None, causal=True
        )
        output = rearrange(output, "(b h) s d -> b s h d", h=self.num_heads)

    attn_output = self.o_proj(output.reshape(bsz, q_len, self.hidden_size))

    return attn_output, None, past_key_value


# Disable the transformation of the attention mask in LlamaModel.forward
# as we handle it in our attention module
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    return attention_mask


def replace_llama_attn_with_flash_attn():
    """
    Replaces the standard attention mechanism in Llama models with Flash Attention for improved performance.
    Works with both Llama2 and Llama3 architectures.
    """
    
    logging.info("Replace Llama Attention with Flash Attention")
    
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        logging.warning(
            "Flash attention is only supported on Ampere or newer GPUs. (CUDA capability >= 8.0)"
        )
        return

    # Monkey patch LlamaModel's attention mechanism to use flash attention
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward
    
    # For Llama3 models, also try to patch MllamaAttention if it exists
    try:
        import transformers.models.mllamaAttention
        transformers.models.mllamaAttention.MllamaAttention.forward = forward
    except (ImportError, AttributeError):
        pass
    
    # Disable attention mask transformation in LlamaModel
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (
        _prepare_decoder_attention_mask
    ) 