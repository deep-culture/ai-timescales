"""Autoregressive transformer generator (Llama / any CausalLM via HuggingFace)."""
from __future__ import annotations

import os
from collections import Counter
from typing import Iterator, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseGenerator, StepResult


class LlamaGenerator(BaseGenerator):
    name = "Llama-3.2-1B-Instruct"

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
        hf_token: str | None = None,
        target_device: str | None = None,
    ):
        self.model_id = model_id
        self.name = model_id.split("/")[-1]
        self.hf_token: str | None = (
            hf_token
            or os.environ.get("HF_TOKEN")
        )
        self._target_device = target_device
        self.model = None
        self.tokenizer = None
        self.device: str = "cpu"
        self._final_ids: list[int] = []
        self._final_input_ids: torch.Tensor | None = None  # full seq for multi-turn
        self._prompt_len: int = 0

    def load(self) -> None:
        n_gpus = torch.cuda.device_count()

        if self._target_device is not None:
            self.device = self._target_device
            device_map = self._target_device
        elif n_gpus > 0:
            self.device = "cuda:0"
            device_map = "auto"
        else:
            self.device = "cpu"
            device_map = "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            token=self.hf_token,
            attn_implementation="eager"
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=self.hf_token,
        )

        if hasattr(self.model, "hf_device_map") and n_gpus > 1:
            dist = Counter(str(d) for d in self.model.hf_device_map.values())
            print(f"[Llama] sharded across {n_gpus} GPU(s): {dict(dist)}")
        else:
            print(f"[Llama] loaded on {self.device}")

    def is_loaded(self) -> bool:
        return self.model is not None

    def encode_chat(self, messages: list[dict]) -> torch.Tensor:
        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        return ids.to(self.device)

    def decode_ids(self, ids: list[int], skip_special: bool = False) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special)

    @torch.no_grad()
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

        self._prompt_len = prompt_ids.shape[1]
        self._final_ids = []
        self._final_input_ids = None

        input_ids = prompt_ids.to(self.device)
        generated: list[int] = []
        eos_id = self.tokenizer.eos_token_id

        for step in range(gen_length):
            output = self.model(input_ids, output_attentions=return_attention)
            next_token_logits = output.logits[:, -1, :].to(self.device)

            if temperature == 0.0:
                next_token_id = int(torch.argmax(next_token_logits, dim=-1).item())
            else:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token_id = int(torch.multinomial(probs, num_samples=1).item())

            generated.append(next_token_id)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)],
                dim=1,
            )

            # Last-layer attention: (1, n_heads, seq, seq) → keep all heads → (n_heads, seq, seq)
            attn: torch.Tensor | None = None
            if return_attention and output.attentions:
                attn = output.attentions[-1][0].detach().cpu()  # (n_heads, T, T)

            decoded: List[str] = [self.decode_ids([tid]) for tid in generated]

            yield StepResult(
                step_index=step,
                token_ids=list(generated),
                decoded_tokens=decoded,
                attention=attn,
                mask_positions=[],
                newly_revealed=[len(generated) - 1],
            )

            if eos_id is not None and next_token_id == eos_id:
                break

        self._final_ids = list(generated)
        self._final_input_ids = input_ids  # full prompt + generation, for multi-turn

    def get_final_text(self) -> str:
        if not self._final_ids:
            return ""
        return self.tokenizer.decode(self._final_ids, skip_special_tokens=True)

    def get_final_token_ids(self) -> list[int]:
        """Full sequence (prompt + generation) without EOS — for multi-turn context."""
        if self._final_input_ids is None:
            return []
        ids = self._final_input_ids[0].cpu().tolist()
        eos_id = self.tokenizer.eos_token_id
        if eos_id and ids and ids[-1] == eos_id:
            ids = ids[:-1]
        return ids

