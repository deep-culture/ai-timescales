"""Abstract base class for text generators (diffusion, autoregressive, etc.)."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import torch


@dataclass
class StepResult:
    """Snapshot produced after a single denoising / decoding step."""
    step_index: int                         # 0-based
    token_ids: List[int]                    # generated token IDs (prompt excluded)
    decoded_tokens: List[str]               # per-token decoded strings
    attention: Optional[torch.Tensor] = None  # (n_heads, seq, seq) or None
    mask_positions: List[int] = field(default_factory=list)  # indices still masked
    newly_revealed: List[int] = field(default_factory=list)  # indices revealed this step
    elapsed_s: float = 0.0                  # seconds since generation start (set by backend)


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

