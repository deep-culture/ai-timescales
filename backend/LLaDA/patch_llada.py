# patch_llada.py
import time
import torch
import types
import torch.nn.functional as F
from typing import List, Optional, Tuple


def _patched_scaled_dot_product_attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """SDPA that exposes attention weights only while capture is enabled.

    When ``self._capture_attn`` is False we delegate to the block's original
    implementation (saved as ``_orig_sdpa`` at patch time), so the inference
    timescale keeps its original fused/flash speed and numerics untouched.
    """
    if not getattr(self, "_capture_attn", False):
        return self._orig_sdpa(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
        )

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
    out = self.attn_out(att)
    # Store the attention sub-layer's residual write (B, T, C) for direct logit
    # attribution. This is exactly the vector added to the residual stream.
    self._attn_output = out
    return out, present


def _record_attn_eigenzeit(module, _args):
    """forward_pre_hook fired the instant the GPU reaches this block.

    We sit on the block (not the attention method) so timing is decoupled from
    the weight-capture path — the same hook would fire on the fused path too. We
    ``record()`` a CUDA timing event into the stream (async — no CPU stall) and
    also stamp a CPU ``perf_counter_ns`` as a fallback for non-CUDA runs. Gated on
    ``_capture_attn`` so the inference timescale pays nothing.
    """
    if not getattr(module, "_capture_attn", False):
        return
    ev = getattr(module, "_attn_start_event", None)
    if ev is not None:
        ev.record()
    module._attn_time_ns = time.perf_counter_ns()


def patch_model(model):
    """
    Patch EVERY transformer block in a loaded LLaDA model so each block exposes
    its attention weights via ``block._attn_weights`` and its Eigenzeit (when the
    GPU executed it) via a per-block CUDA timing event recorded by a forward
    pre-hook.

    This is required for the second ("attention") timescale, which sonifies the
    scores of every layer (32 layers × 32 heads). The manual attention path is
    slower than fused/flash attention, but it is the only way to read the
    weights back out — and we touch nothing else in the forward pass.
    """
    use_events = torch.cuda.is_available()
    blocks = model.model.transformer.blocks
    for blk in blocks:
        # Preserve the original SDPA so the non-capture path stays fast/faithful.
        blk._orig_sdpa = blk._scaled_dot_product_attention
        blk._scaled_dot_product_attention = types.MethodType(
            _patched_scaled_dot_product_attention, blk
        )
        blk.attention = types.MethodType(_patched_attention, blk)
        blk._attn_weights = None
        blk._attn_output = None
        blk._attn_time_ns = 0
        # GPU timing marker (reused/re-recorded each step). None on CPU/MPS.
        blk._attn_start_event = (
            torch.cuda.Event(enable_timing=True) if use_events else None
        )
        blk.register_forward_pre_hook(_record_attn_eigenzeit)
        blk._capture_attn = False
    print(f"Patched all {len(blocks)} blocks for per-layer attention capture.")
    return model


def set_capture(model, flag: bool) -> None:
    """Enable/disable attention-weight capture on every block at once.

    Capture is only turned on for the attention timescale; the inference
    timescale leaves it off and pays no extra cost.
    """
    for blk in model.model.transformer.blocks:
        blk._capture_attn = bool(flag)


def collect_layer_attention(
    model,
) -> Tuple[Optional[torch.Tensor], List[int]]:
    """
    Gather every block's stored attention weights into one tensor and read back
    the per-layer Eigenzeit from its CUDA timing event, then release the GPU
    copies.

    Returns ``(attn, timings_ns)`` where ``attn`` is ``(n_layers, n_heads, T, T)``
    on the CPU (float32, batch index 0) and ``timings_ns`` is one ns offset per
    layer (GPU execution time relative to layer 0; CPU fallback when no CUDA).
    Returns ``(None, [])`` if any block has no weights stored (e.g. a
    flash-attention path that can't expose them).
    """
    from inference.base import gpu_event_offsets_ns  # local: avoid import cycle

    blocks = model.model.transformer.blocks
    weights: List[torch.Tensor] = []
    events: List[Optional[torch.cuda.Event]] = []
    fallback_ns: List[int] = []
    for blk in blocks:
        w = getattr(blk, "_attn_weights", None)
        if w is None:
            return None, []
        # w: (B, n_heads, T, T) — keep the first batch item (conditional pass).
        weights.append(w[0].detach().to("cpu", torch.float32))
        events.append(getattr(blk, "_attn_start_event", None))
        fallback_ns.append(int(getattr(blk, "_attn_time_ns", 0)))
        blk._attn_weights = None  # free the GPU copy now that it's been read
    timings = gpu_event_offsets_ns(events, fallback_ns)
    return torch.stack(weights, dim=0), timings


def collect_layer_outputs(model) -> Optional[torch.Tensor]:
    """
    Stack every block's attention residual write (``_attn_output``) into one
    tensor ``(n_layers, T, C)`` on the model's device (batch index 0), for direct
    logit attribution, then release the per-block copies. Returns ``None`` if any
    block has nothing stored.
    """
    blocks = model.model.transformer.blocks
    outs: List[torch.Tensor] = []
    for blk in blocks:
        o = getattr(blk, "_attn_output", None)
        if o is None:
            return None
        outs.append(o[0].detach())  # (T, C), first batch item
        blk._attn_output = None
    return torch.stack(outs, dim=0)


def clear_layer_buffers(model) -> None:
    """Drop any stored per-block attention weights/outputs to free GPU memory."""
    for blk in model.model.transformer.blocks:
        blk._attn_weights = None
        blk._attn_output = None
