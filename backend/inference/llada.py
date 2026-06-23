"""LLaDA diffusion-language-model generator."""
from __future__ import annotations

from collections import Counter
from typing import Iterator, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from LLaDA.patch_llada import (
    collect_layer_attention,
    collect_layer_outputs,
    patch_model,
    set_capture,
)
from .base import BaseGenerator, StepResult, rms_logit_attribution

MASK_ID = 126336
EOS_ID = 126081


def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    return logits.exp() / ((-torch.log(noise)) ** temperature)


def _get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num[i, : remainder[i]] += 1
    return num


class LLaDAGenerator(BaseGenerator):
    name = "LLaDA-8B-Instruct"

    def __init__(
        self,
        model_id: str = "GSAI-ML/LLaDA-8B-Instruct",
        target_device: str | None = None,
    ):
        self.model_id = model_id
        # None  → Accelerate picks automatically (device_map="auto")
        # "cuda:0" etc. → entire model on that one device
        self._target_device = target_device
        self.model = None
        self.tokenizer = None
        self.device: str = "cpu"
        self._final_x: torch.Tensor | None = None
        self._prompt_len: int = 0

    def load(self) -> None:
        n_gpus = torch.cuda.device_count()

        if self._target_device is not None:
            self.device = self._target_device
        elif n_gpus > 0:
            self.device = "cuda:1"
        else:
            self.device = "cpu"

        # NOTE: We deliberately do NOT pass device_map here.
        # LLaDA uses trust_remote_code and its LLaDAModelLM class does not
        # implement `all_tied_weights_keys`, which transformers >= 4.45 requires
        # inside infer_auto_device_map. LLaDA-8B (~16 GB bfloat16) fits on a
        # single A10 (24 GB), so a plain .to(device) is sufficient.
        self.model = (
            AutoModel.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            .to(self.device)
            .eval()
        )
        self.model = patch_model(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        print(f"[LLaDA] loaded on {self.device}")


    def is_loaded(self) -> bool:
        return self.model is not None

    def encode_chat(self, messages: list[dict]) -> torch.Tensor:
        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        ids = self.tokenizer(text)["input_ids"]
        return torch.tensor(ids, device=self.device).unsqueeze(0)

    def decode_ids(self, ids: list[int], skip_special: bool = False) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special)

    def free_memory(self) -> None:
        """Delete cached tensors and flush the CUDA allocator."""
        self._final_x = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _setup_capture(self, return_attention: bool):
        """Enable per-layer attention capture + hook the final RMSNorm (for DLA).

        Returns the ``ln_f`` module so the per-step body can read its weight/eps.
        A no-op fast path when ``return_attention`` is False.
        """
        set_capture(self.model, return_attention)
        # Remove any stale hook first (a previous run may have been cancelled).
        prev_hook = getattr(self, "_dla_hook", None)
        if prev_hook is not None:
            prev_hook.remove()
        self._dla_hook = None
        self._final_resid = None
        ln_f = self.model.model.transformer.ln_f
        def _ln_f_pre_hook(_m, inp):
            self._final_resid = inp[0].detach()
        if return_attention:
            self._dla_hook = ln_f.register_forward_pre_hook(_ln_f_pre_hook)
        return ln_f

    def _teardown_capture(self) -> None:
        """Disable capture and remove the DLA hook (back to the fast path)."""
        set_capture(self.model, False)
        if self._dla_hook is not None:
            self._dla_hook.remove()
            self._dla_hook = None
        self._final_resid = None

    @torch.no_grad()
    def _denoise_step(
        self,
        x: torch.Tensor,
        global_step: int,
        *,
        steps_per_block: int,
        block_length: int,
        prompt_len: int,
        gen_length: int,
        prompt_index: torch.Tensor,
        temperature: float,
        cfg_scale: float,
        remasking: str,
        return_attention: bool,
        echo_head_indices: list[int] | None,
        ln_f,
    ) -> StepResult:
        """Run ONE denoising step in place on ``x`` and return its ``StepResult``.

        Stateless w.r.t. prior steps: everything needed is ``x`` (the current
        masked sequence) and ``global_step``. The reveal schedule is recomputed
        from ``block_length``/``steps_per_block`` — each block is fully masked
        when first entered (gen positions start masked and blocks resolve
        left→right), so it depends only on the sizes, not on the live ``x``.
        That is what lets a single step be replayed from a client-carried ``x``.
        """
        num_block = global_step // steps_per_block
        i = global_step % steps_per_block
        blk_end = prompt_len + (num_block + 1) * block_length

        # Block is fully masked on entry → schedule depends only on sizes.
        full_block_mask = torch.ones(
            (x.shape[0], block_length), dtype=torch.bool, device=x.device
        )
        num_transfer = _get_num_transfer_tokens(full_block_mask, steps_per_block)

        mask_index = x == MASK_ID
        if cfg_scale > 0.0:
            un_x = x.clone()
            un_x[prompt_index] = MASK_ID
            x_ = torch.cat([x, un_x], dim=0)
            logits = self.model(x_).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = self.model(x).logits

        # With device_map the output may be on the last layer's GPU; normalise
        # back to self.device so all subsequent ops are on the same device as x.
        logits = logits.to(self.device)

        # ── attention capture ────────────────────────────────────────────
        # Every block stores its weights each forward (patch_model). We capture
        # the full per-layer "echo" on EVERY step so the echo playback can
        # replay the step the user is looking at. To keep that affordable, the
        # per-layer echo is pre-averaged over the requested head selection on
        # the GPU (→ (L,T,T)). We still keep the last layer's per-head weights
        # separately for the live blur view + the interactive head selector.
        attn: torch.Tensor | None = None
        attn_layers: torch.Tensor | None = None
        layer_timings: list[int] = []
        attention_dla: torch.Tensor | None = None
        if return_attention:
            blocks = self.model.model.transformer.blocks
            # Live merged view: last layer, per-head (read before collect frees).
            last_w = getattr(blocks[-1], "_attn_weights", None)
            if last_w is not None:
                attn = last_w[0].detach().to("cpu", torch.float32)  # (H,T,T)
            # Per-step echoes: every layer, head-reduced → (L,T,T).
            attn_layers, layer_timings = collect_layer_attention(
                self.model, head_indices=echo_head_indices
            )
            # ── direct logit attribution per layer (echo "heartbeat") ──
            outs = collect_layer_outputs(self.model)   # (L, T, C) on device
            if outs is not None and self._final_resid is not None:
                gen = slice(prompt_len, prompt_len + gen_length)
                target_ids = torch.argmax(logits[0, gen], dim=-1)  # (gen_len,)
                attention_dla = rms_logit_attribution(
                    outs[:, gen, :],               # (L, gen_len, C)
                    self._final_resid[0, gen],     # (gen_len, C)
                    ln_f.weight,
                    float(getattr(ln_f, "eps", 1e-5)),
                    self.model.model.transformer.ff_out.weight,
                    target_ids,
                ).cpu()                            # (L, gen_len)

        logits_noisy = _add_gumbel_noise(logits, temperature)
        x0 = torch.argmax(logits_noisy, dim=-1)

        if remasking == "low_confidence":
            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=x0.unsqueeze(-1)), -1
            )
        elif remasking == "random":
            x0_p = torch.rand(x0.shape, device=x0.device)
        else:
            raise NotImplementedError(remasking)

        x0_p[:, blk_end:] = -np.inf
        old_x = x.clone()
        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)

        transfer_index = torch.zeros_like(x0, dtype=torch.bool)
        for j in range(confidence.shape[0]):
            _, sel = torch.topk(confidence[j], k=num_transfer[j, i])
            transfer_index[j, sel] = True
        x[transfer_index] = x0[transfer_index]

        gen_ids = x[0, prompt_len:].cpu().tolist()
        old_gen = old_x[0, prompt_len:].cpu().tolist()

        decoded: List[str] = []
        masks: List[int] = []
        revealed: List[int] = []
        for idx, tid in enumerate(gen_ids):
            decoded.append(self.decode_ids([tid]))
            if tid == MASK_ID:
                masks.append(idx)
            elif old_gen[idx] == MASK_ID:
                revealed.append(idx)

        return StepResult(
            step_index=global_step,
            token_ids=gen_ids,
            decoded_tokens=decoded,
            attention=attn,
            mask_positions=masks,
            newly_revealed=revealed,
            attention_layers=attn_layers,
            layer_timings_ns=layer_timings,
            attention_dla=attention_dla,
        )

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
        echo_head_indices: list[int] | None = None,
    ) -> Iterator[StepResult]:
        self.free_memory()  # release previous run's tensors before allocating new ones
        ln_f = self._setup_capture(return_attention)

        prompt_len = prompt_ids.shape[1]
        self._prompt_len = prompt_len

        x = torch.full(
            (prompt_ids.shape[0], prompt_len + gen_length),
            MASK_ID, dtype=torch.long, device=self.device,
        )
        x[:, :prompt_len] = prompt_ids.clone()
        prompt_index = torch.zeros_like(x, dtype=torch.bool)
        prompt_index[:, :prompt_len] = True

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks

        for global_step in range(steps):
            yield self._denoise_step(
                x, global_step,
                steps_per_block=steps_per_block, block_length=block_length,
                prompt_len=prompt_len, gen_length=gen_length,
                prompt_index=prompt_index, temperature=temperature,
                cfg_scale=cfg_scale, remasking=remasking,
                return_attention=return_attention,
                echo_head_indices=echo_head_indices, ln_f=ln_f,
            )

        self._teardown_capture()
        self._final_x = x

    @torch.no_grad()
    def generate_single_step(
        self,
        prompt_ids: torch.Tensor,
        gen_ids: list[int],
        global_step: int,
        gen_length: int = 64,
        steps: int = 32,
        block_length: int = 32,
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        remasking: str = "low_confidence",
        return_attention: bool = False,
        echo_head_indices: list[int] | None = None,
    ) -> Iterator[StepResult]:
        """Run exactly ONE denoising step — the one at ``global_step`` — resuming
        from ``gen_ids`` (the current generated portion, ``MASK_ID`` where still
        unrevealed; empty/short → treated as all-masked).

        Stateless: the client carries the sequence between presses, so no KV
        cache or server-side session is needed (LLaDA recomputes the full
        forward each step regardless). Yields a single ``StepResult`` so it
        shares the SSE streaming path with ``generate_steps``.
        """
        self.free_memory()
        ln_f = self._setup_capture(return_attention)

        prompt_len = prompt_ids.shape[1]
        self._prompt_len = prompt_len

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks
        global_step = max(0, min(int(global_step), steps - 1))

        # Sanity check: any step past the first must resume from a partially
        # revealed sequence. An empty/all-masked resume at start_step > 0 means
        # the client lost the carried sequence — we don't fail (the step still
        # runs from an all-masked state), but flag it so the cause is visible.
        if global_step > 0 and (
            not gen_ids or all(t == MASK_ID for t in gen_ids)
        ):
            reason = "empty" if not gen_ids else "all-masked"
            print(
                f"[LLaDA] single-step warning: start_step={global_step} but "
                f"resume_gen_ids is {reason} — the client likely lost the "
                f"carried sequence; this step runs as if from a fresh state.",
                flush=True,
            )

        # Rebuild x = prompt ++ carried generated portion (MASK where unrevealed).
        gen = torch.full(
            (prompt_ids.shape[0], gen_length), MASK_ID,
            dtype=torch.long, device=self.device,
        )
        if gen_ids:
            vals = torch.tensor(
                gen_ids[:gen_length], dtype=torch.long, device=self.device
            )
            gen[0, : vals.shape[0]] = vals
        x = torch.cat([prompt_ids.to(self.device), gen], dim=1)
        prompt_index = torch.zeros_like(x, dtype=torch.bool)
        prompt_index[:, :prompt_len] = True

        sr = self._denoise_step(
            x, global_step,
            steps_per_block=steps_per_block, block_length=block_length,
            prompt_len=prompt_len, gen_length=gen_length,
            prompt_index=prompt_index, temperature=temperature,
            cfg_scale=cfg_scale, remasking=remasking,
            return_attention=return_attention,
            echo_head_indices=echo_head_indices, ln_f=ln_f,
        )

        self._teardown_capture()
        self._final_x = x
        yield sr

    def get_final_text(self) -> str:
        if self._final_x is None:
            return ""
        return self.tokenizer.batch_decode(
            self._final_x[:, self._prompt_len:], skip_special_tokens=True
        )[0]

    def get_final_token_ids(self) -> list[int]:
        """Full sequence (prompt + generation) without EOS — for multi-turn context."""
        if self._final_x is None:
            return []
        x = self._final_x[0]
        return x[x != EOS_ID].cpu().tolist()

    def strip_eos(self, prompt: torch.Tensor) -> torch.Tensor:
        return prompt[prompt != EOS_ID].unsqueeze(0)
