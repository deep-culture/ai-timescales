"""LLaDA diffusion-language-model generator."""
from __future__ import annotations

from collections import Counter
from typing import Iterator, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from LLaDA.patch_llada import patch_model
from .base import BaseGenerator, StepResult

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
            self.device = "cuda:0"
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
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
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
    ) -> Iterator[StepResult]:

        prompt_len = prompt_ids.shape[1]
        self._prompt_len = prompt_len

        x = torch.full(
            (prompt_ids.shape[0], prompt_len + gen_length),
            MASK_ID, dtype=torch.long, device=self.device,
        )
        x[:, :prompt_len] = prompt_ids.clone()
        prompt_index = x != MASK_ID

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks
        global_step = 0

        for num_block in range(num_blocks):
            blk_start = prompt_len + num_block * block_length
            blk_end   = prompt_len + (num_block + 1) * block_length
            block_mask_index = x[:, blk_start:blk_end] == MASK_ID
            num_transfer = _get_num_transfer_tokens(block_mask_index, steps_per_block)

            for i in range(steps_per_block):
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

                # With device_map the output may be on the last layer's GPU;
                # normalise back to self.device so all subsequent ops are on
                # the same device as x.
                logits = logits.to(self.device)

                attn = self.model.model.transformer.blocks[-1]._attn_weights
                attn = attn.detach().cpu() if attn is not None else None

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

                yield StepResult(
                    step_index=global_step,
                    token_ids=gen_ids,
                    decoded_tokens=decoded,
                    attention=attn,
                    mask_positions=masks,
                    newly_revealed=revealed,
                )
                global_step += 1

        self._final_x = x

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
