"""
AI-Timescales – FastAPI inference server.
...
"""
from __future__ import annotations

import asyncio
import json
import threading

from dotenv import load_dotenv
from pathlib import Path
import os

_HERE = Path(__file__).parent.resolve()
load_dotenv(_HERE / ".env")

if not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = str(_HERE / "huggingface")

print(f"[server] HF_HOME = {os.environ['HF_HOME']}", flush=True)

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from inference import LLaDAGenerator, LlamaGenerator
from inference.base import BaseGenerator
from inference.llada import EOS_ID

# ── global cancellation flag (one generation at a time) ──────────────────
_cancel = threading.Event()

# ── model registry (populated in lifespan) ───────────────────────────────
_GENERATORS: dict[str, BaseGenerator] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the model registry based on available GPUs, then load every model."""
    n_gpus = torch.cuda.device_count()
    print(f"[server] {n_gpus} CUDA GPU(s) detected", flush=True)

    if n_gpus >= 2:
        # Dedicated GPU per model: LLaDA on cuda:0, Llama on cuda:1
        registry: dict[str, BaseGenerator] = {
            "LLaDA-8B-Instruct":              LLaDAGenerator(target_device="cuda:0"),
            "Llama-3.2-1B-Instruct": LlamaGenerator(target_device="cuda:1"),
        }
    else:
        # Single GPU or CPU – Accelerate distributes each model with device_map="auto"
        registry = {
            "LLaDA-8B-Instruct":              LLaDAGenerator(),
            "Llama-3.2-1B-Instruct":          LlamaGenerator(),
        }

    loop = asyncio.get_running_loop()
    for name, gen in registry.items():
        print(f"[server] loading {name}…", flush=True)
        await loop.run_in_executor(None, gen.load)
        print(f"[server] {name} ready ✓", flush=True)
        _GENERATORS[name] = gen

    yield
    # nothing to clean up on shutdown


app = FastAPI(title="AI Timescales Inference Server", lifespan=lifespan)

# ── request schema ─────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    model: str = "LLaDA-8B-Instruct"
    messages: list[dict] = Field(default_factory=list)
    prev_token_ids: list[int] = Field(
        default_factory=list,
        description="Full accumulated prompt token IDs from all previous turns (multi-turn).",
    )
    gen_length: int = 64
    steps: int = 8
    block_length: int = 32
    temperature: float = 0.0
    cfg_scale: float = 0.0
    remasking: str = "low_confidence"


# ── SSE helpers ─────────────────────────────────────────────────────────────
def _sse(event: str, data: dict) -> str:
    """Format a single SSE message (event + data + blank-line delimiter)."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── endpoints ───────────────────────────────────────────────────────────────
@app.get("/status")
async def server_status() -> dict:
    """Richer status: includes which models are loaded and ready."""
    return {
        "status": "ok",
        "models": {name: "ready" for name in _GENERATORS},
    }

@app.get("/models")
async def list_models() -> list[str]:
    return list(_GENERATORS.keys())


@app.post("/stop")
async def stop_generation() -> dict:
    """Signal the active generation to stop after its current step."""
    _cancel.set()
    return {"status": "stopping"}


@app.post("/generate")
async def generate(req: GenerateRequest) -> StreamingResponse:
    if req.model not in _GENERATORS:
        raise HTTPException(status_code=404, detail=f"Unknown model '{req.model}'")

    gen = _GENERATORS[req.model]
    _cancel.clear()  # reset flag for this new generation

    # ── build the prompt tensor server-side ────────────────────────────────
    new_ids = gen.encode_chat(req.messages)   # (1, L) – always includes BOS
    if req.prev_token_ids:
        # Multi-turn: prepend accumulated context, strip BOS from new turn
        prev = torch.tensor([req.prev_token_ids], device=gen.device)
        prompt = torch.cat([prev, new_ids[:, 1:]], dim=1)
    else:
        prompt = new_ids

    async def event_stream() -> AsyncGenerator[str, None]:
        loop = asyncio.get_running_loop()
        # Queue items: StepResult | Exception | None (None = sentinel / done)
        q: asyncio.Queue = asyncio.Queue()

        def _run_gen() -> None:
            try:
                for sr in gen.generate_steps(
                    prompt,
                    gen_length=req.gen_length,
                    steps=req.steps,
                    block_length=req.block_length,
                    temperature=req.temperature,
                    cfg_scale=req.cfg_scale,
                    remasking=req.remasking,
                ):
                    if _cancel.is_set():
                        break   # stop between steps; current CUDA op finishes cleanly
                    asyncio.run_coroutine_threadsafe(q.put(sr), loop)
            except Exception as exc:  # noqa: BLE001
                asyncio.run_coroutine_threadsafe(q.put(exc), loop)
            finally:
                asyncio.run_coroutine_threadsafe(q.put(None), loop)

        loop.run_in_executor(None, _run_gen)

        while True:
            item = await q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                yield _sse("error", {"message": str(item)})
                return
            # item is a StepResult
            yield _sse("step", {
                "step_index":     item.step_index,
                "token_ids":      item.token_ids,
                "decoded_tokens": item.decoded_tokens,
                "mask_positions": item.mask_positions,
                "newly_revealed": item.newly_revealed,
            })

        # ── done or cancelled ──────────────────────────────────────────────
        if _cancel.is_set():
            yield _sse("cancelled", {"message": "Generation stopped by user"})
            return

        yield _sse("done", {
            "final_token_ids": gen.get_final_token_ids(),
            "final_text":      gen.get_final_text(),
        })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

