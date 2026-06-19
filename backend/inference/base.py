"""Abstract base class for text generators (diffusion, autoregressive, etc.)."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import torch


def gpu_event_offsets_ns(
    events: List[Optional["torch.cuda.Event"]],
    fallback_ns: List[int],
) -> List[int]:
    """Per-layer Eigenzeit as *when the GPU actually executed* each layer.

    ``events`` holds one CUDA timing event per layer, each ``record()``-ed into
    the stream at the instant its layer began. We synchronize once here (the only
    blocking call — and it happens after the forward pass, never mid-pass) and
    then read the GPU-measured gaps, returned as nanosecond offsets relative to
    the first layer (so layer 0 is always 0).

    Falls back to the CPU ``perf_counter_ns`` timestamps in ``fallback_ns`` when
    CUDA events are unavailable (CPU/MPS runs) or were never recorded — the wire
    format (ns offsets per layer) is identical either way.
    """
    if torch.cuda.is_available() and events and all(e is not None for e in events):
        try:
            torch.cuda.synchronize()
            t0 = events[0]
            # elapsed_time is GPU-measured milliseconds → nanoseconds.
            return [int(round(t0.elapsed_time(e) * 1e6)) for e in events]
        except (RuntimeError, ValueError):
            pass  # event not recorded / cross-device — fall through to CPU times
    return list(fallback_ns)


def rms_logit_attribution(
    attn_outputs: torch.Tensor,   # (n_layers, P, C) per-layer attention residual writes
    x_final: torch.Tensor,        # (P, C) full residual entering the final RMSNorm
    gamma: torch.Tensor,          # (C,) final RMSNorm weight
    eps: float,                   # RMSNorm epsilon
    unembed_weight: torch.Tensor, # (V, C) unembedding / lm_head weight
    target_ids: torch.Tensor,     # (P,) token id to attribute toward at each position
) -> torch.Tensor:
    """
    Direct Logit Attribution for RMSNorm models.

    Both LLaDA and Llama end in RMSNorm + unembedding, and a softmax over the
    output is shift-invariant, so the contribution of a single residual-stream
    component v to a token's logit is, with the final-LayerNorm scale frozen,

        dla = (gamma ⊙ v / rms(x_final)) · W_U[token]

    where rms(x_final) uses the *full* final residual at that position. We apply
    this to each attention layer's residual write to measure how much that layer
    pushes the output toward the chosen token. Returns ``(n_layers, P)`` (signed —
    a layer can also push *against* the token).
    """
    dev = unembed_weight.device
    a = attn_outputs.float().to(dev)                      # (L, P, C)
    xf = x_final.float().to(dev)                          # (P, C)
    g = gamma.float().to(dev)                             # (C,)
    rms = torch.sqrt(xf.pow(2).mean(-1) + eps)            # (P,)
    scale = g.unsqueeze(0) / rms.unsqueeze(1)             # (P, C)
    normed = a * scale.unsqueeze(0)                       # (L, P, C)
    W = unembed_weight.index_select(0, target_ids.to(dev)).float()  # (P, C)
    return torch.einsum("lpc,pc->lp", normed, W)          # (L, P)


@dataclass
class StepResult:
    """Snapshot produced after a single denoising / decoding step."""
    step_index: int                         # 0-based
    token_ids: List[int]                    # generated token IDs (prompt excluded)
    decoded_tokens: List[str]               # per-token decoded strings
    attention: Optional[torch.Tensor] = None  # last-layer (n_heads, seq, seq) or None
    mask_positions: List[int] = field(default_factory=list)  # indices still masked
    newly_revealed: List[int] = field(default_factory=list)  # indices revealed this step
    elapsed_s: float = 0.0                  # seconds since generation start (set by backend)

    # ── second timescale: per-layer attention "echoes" ──────────────────────
    # Populated only on the *final* step of a sequence (the layer playback runs
    # once generation is done). ``attention_layers`` stacks every transformer
    # layer's attention, ``layer_timings_ns`` records, via CUDA timing events,
    # *when the GPU actually executed* each layer relative to the first (ns
    # offsets, layer 0 = 0) — the "Eigenzeit" of the embeddings hitting each
    # layer. Falls back to CPU perf_counter_ns offsets when CUDA is unavailable.
    attention_layers: Optional[torch.Tensor] = None  # (n_layers, n_heads, seq, seq)
    layer_timings_ns: List[int] = field(default_factory=list)  # one ns offset per layer
    # Direct logit attribution of each attention layer to the chosen token(s):
    #   AR  → shape (n_layers,)            (the single next-token prediction)
    #   DLM → shape (n_layers, gen_len)    (one column per generated position)
    attention_dla: Optional[torch.Tensor] = None


class BaseGenerator(abc.ABC):
    """
    Unified interface for any HuggingFace-compatible text generator.

    Subclass this for each model family (LLaDA, GPT-style, etc.).
    """

    name: str = "base"

    # ── lifecycle ────────────────────────────────────────────────────────
    @abc.abstractmethod
    def load(self) -> None:
        """Load model + tokenizer onto the target device."""

    @abc.abstractmethod
    def is_loaded(self) -> bool:
        ...

    # ── tokenisation helpers ─────────────────────────────────────────────
    @abc.abstractmethod
    def encode_chat(self, messages: list[dict]) -> torch.Tensor:
        """Apply chat template + tokenise → (1, L) tensor on device."""

    @abc.abstractmethod
    def decode_ids(self, ids: list[int], skip_special: bool = False) -> str:
        """Decode a list of token IDs to a string."""

    # ── generation ───────────────────────────────────────────────────────
    @abc.abstractmethod
    def generate_steps(
        self,
        prompt_ids: torch.Tensor,
        gen_length: int = 64,
        steps: int = 32,
        block_length: int = 32,
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        remasking: str = "low_confidence",
        return_attention: bool = False,
    ) -> Iterator[StepResult]:
        """
        Yield one ``StepResult`` per denoising / decoding step.

        For diffusion models each yield is a denoising step.
        For autoregressive models each yield could be a single token.
        """

    @abc.abstractmethod
    def get_final_text(self) -> str:
        """Return the fully decoded answer after generation has finished."""

    @abc.abstractmethod
    def get_final_token_ids(self) -> list[int]:
        """Full token ID sequence (prompt + generation, EOS stripped) for multi-turn."""

    @abc.abstractmethod
    def free_memory(self) -> None:
        """Release any cached tensors and free the CUDA allocator cache."""
