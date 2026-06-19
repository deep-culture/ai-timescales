"""Autoregressive transformer generator (Llama / any CausalLM via HuggingFace)."""
from __future__ import annotations

import os
import time
from collections import Counter
from typing import Iterator, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseGenerator, StepResult, rms_logit_attribution


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
        self._layer_entry_ns: dict[int, int] = {}  # per-layer Eigenzeit, reset each forward

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
            token=self.hf_token
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

    def free_memory(self) -> None:
        """Delete cached tensors and flush the CUDA allocator."""
        self._final_input_ids = None
        self._final_ids = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _register_capture_hooks(self) -> list:
        """
        Attach hooks needed for the attention timescale, all read-only w.r.t. the
        forward pass:
          • per-layer pre-hook → Eigenzeit timestamp (when embeddings enter it)
          • per-layer post-hook → the attention sub-layer's residual write at the
            last position (for direct logit attribution)
          • a pre-hook on the final RMSNorm → the residual entering it (for DLA)
        Returns the hook handles to remove later.
        """
        handles = []
        for idx, layer in enumerate(self.model.model.layers):
            def _make_timing(i: int):
                def _hook(_module, _inp):
                    self._layer_entry_ns[i] = time.perf_counter_ns()
                return _hook
            def _make_output(i: int):
                def _hook(_module, _inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    self._layer_attn_out[i] = o[:, -1, :].detach()[0]  # (C,)
                return _hook
            handles.append(layer.self_attn.register_forward_pre_hook(_make_timing(idx)))
            handles.append(layer.self_attn.register_forward_hook(_make_output(idx)))

        def _norm_pre(_module, inp):
            self._final_resid = inp[0][:, -1, :].detach()[0]  # (C,)
        handles.append(self.model.model.norm.register_forward_pre_hook(_norm_pre))
        return handles

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
        self.free_memory()  # release previous run's tensors before allocating new ones

        self._prompt_len = prompt_ids.shape[1]
        self._final_ids = []
        self._final_input_ids = None

        input_ids = prompt_ids.to(self.device)
        generated: list[int] = []
        eos_id = self.tokenizer.eos_token_id

        capture_hooks = self._register_capture_hooks() if return_attention else []
        try:
            for step in range(gen_length):
                self._layer_entry_ns = {}   # reset per-layer timestamps for this forward
                self._layer_attn_out = {}   # reset per-layer attention residual writes
                self._final_resid = None    # reset final residual
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

                is_eos = eos_id is not None and next_token_id == eos_id
                is_last = (step == gen_length - 1) or is_eos

                # Per-step we keep the last layer for the live blur view. On the
                # final step we additionally stack every layer (n_layers, n_heads,
                # T, T) plus its per-layer Eigenzeit timestamp for the playback.
                attn: torch.Tensor | None = None
                attn_layers: torch.Tensor | None = None
                layer_timings: list[int] = []
                attention_dla: torch.Tensor | None = None
                if return_attention and output.attentions:
                    attn = output.attentions[-1][0].detach().to("cpu", torch.float32)
                    if is_last:
                        n_layers = len(output.attentions)
                        attn_layers = torch.stack(
                            [a[0].detach().to("cpu", torch.float32) for a in output.attentions],
                            dim=0,
                        )
                        layer_timings = [
                            self._layer_entry_ns.get(i, 0) for i in range(n_layers)
                        ]
                        # ── direct logit attribution for the generated token ──
                        if (self._final_resid is not None
                                and len(self._layer_attn_out) == n_layers):
                            outs = torch.stack(
                                [self._layer_attn_out[i] for i in range(n_layers)], dim=0
                            )  # (L, C)
                            target = torch.tensor([next_token_id], device=self.device)
                            attention_dla = rms_logit_attribution(
                                outs.unsqueeze(1),                  # (L, 1, C)
                                self._final_resid.unsqueeze(0),     # (1, C)
                                self.model.model.norm.weight,
                                float(self.model.model.norm.variance_epsilon),
                                self.model.lm_head.weight,
                                target,
                            ).squeeze(1).cpu()                      # (L,)

                decoded: List[str] = [self.decode_ids([tid]) for tid in generated]

                yield StepResult(
                    step_index=step,
                    token_ids=list(generated),
                    decoded_tokens=decoded,
                    attention=attn,
                    mask_positions=[],
                    newly_revealed=[len(generated) - 1],
                    attention_layers=attn_layers,
                    layer_timings_ns=layer_timings,
                    attention_dla=attention_dla,
                )

                if is_eos:
                    break
        finally:
            for h in capture_hooks:
                h.remove()

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

