# patch_llada.py
import torch
import types
import torch.nn.functional as F
from typing import Optional, Tuple


def _patched_scaled_dot_product_attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Manual SDPA that also returns attention weights."""
    if self.flash_attn_func is not None and attn_mask is None:
        # Flash attention path - can't extract weights
        r = self.flash_attn_func(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=dropout_p, causal=False
        )
        return r.transpose(1, 2), None

    # Manual path - exposes attention weights
    assert k.size(1) == v.size(1)
    num_kv_heads = k.size(1)
    num_q_heads = q.size(1)
    if num_q_heads != num_kv_heads:
        assert num_q_heads % num_kv_heads == 0
        k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
        v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)

    scale = q.size(-1) ** -0.5
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)
    return torch.matmul(attn_weights, v), attn_weights  # (B,nh,T,hs), (B,nh,T,T)


def _patched_attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_bias: Optional[torch.Tensor] = None,
    layer_past=None,
    use_cache: bool = False,
):
    B, T, C = q.size()
    dtype = k.dtype

    if self.q_norm is not None and self.k_norm is not None:
        q = self.q_norm(q).to(dtype=dtype)
        k = self.k_norm(k).to(dtype=dtype)

    q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
    k = k.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
    v = v.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

    if layer_past is not None:
        past_key, past_value = layer_past
        k = torch.cat((past_key, k), dim=-2)
        v = torch.cat((past_value, v), dim=-2)

    present = (k, v) if use_cache else None
    query_len, key_len = q.shape[-2], k.shape[-2]

    if self.config.rope:
        q, k = self.rotary_emb(q, k)

    if attention_bias is not None:
        attention_bias = self._cast_attn_bias(
            attention_bias[:, :, key_len - query_len: key_len, :key_len], dtype
        )

    att, attn_weights = self._scaled_dot_product_attention(
        q, k, v,
        attn_mask=attention_bias,
        dropout_p=0.0 if not self.training else self.config.attention_dropout,
        is_causal=False,
    )

    # Store for external access: shape (B, n_heads, T, T)
    self._attn_weights = attn_weights

    att = att.transpose(1, 2).contiguous().view(B, T, C)
    return self.attn_out(att), present


def patch_model(model):
    """
    Patch all transformer blocks in a loaded LLaDA model to expose
    attention weights via block._attn_weights after each forward pass.
    """
    for block in model.model.transformer.blocks:
        block._scaled_dot_product_attention = types.MethodType(
            _patched_scaled_dot_product_attention, block
        )
        block.attention = types.MethodType(_patched_attention, block)
    print(f"Patched {len(model.model.transformer.blocks)} blocks.")
    return model
