"""
AI-Timescales – FastAPI inference server.
...
"""
from __future__ import annotations

import asyncio
import base64 as _base64mod
import io
import json
import threading
import wave

from dotenv import load_dotenv
from pathlib import Path
import os

HERE = Path(__file__).parent.resolve()
load_dotenv(HERE / ".env")

if not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = str(HERE / "huggingface")

# ── FastAPI imports first, before anything that uses them ─────────────────
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field


print(f"[server] HF_HOME = {os.environ['HF_HOME']}", flush=True)


from inference import LLaDAGenerator, LlamaGenerator
from inference.base import BaseGenerator
from inference.llada import EOS_ID

# ── global cancellation flag (one generation at a time) ──────────────────
_cancel = threading.Event()

# ── model registry (populated in lifespan) ───────────────────────────────
_GENERATORS: dict[str, BaseGenerator] = {}

# ── TTS pipeline (Kokoro, loaded in lifespan) ────────────────────────────
_tts_pipeline = None
_tts_lock = threading.Lock()
_tts_cache: dict[tuple, bytes] = {}   # (text, speed) → WAV bytes


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the model registry based on available GPUs, then load every model."""
    global _tts_pipeline

    # Load Kokoro TTS (CPU, fast)
    print("[server] loading Kokoro TTS…", flush=True)
    try:
        from kokoro import KPipeline
        _tts_pipeline = KPipeline(lang_code="a")
        print("[server] Kokoro TTS ready ✓", flush=True)
    except Exception as exc:
        print(f"[server] Kokoro TTS unavailable: {exc}", flush=True)

    n_gpus = torch.cuda.device_count()
    print(f"[server] {n_gpus} CUDA GPU(s) detected", flush=True)

    skip_ar        = os.environ.get("DO_NOT_LOAD_AR", "false").strip().lower() == "true"
    skip_diffusion = os.environ.get("DO_NOT_LOAD_DIFFUSION", "false").strip().lower() == "true"

    if n_gpus >= 2:
        registry: dict[str, BaseGenerator] = {}
        if not skip_diffusion:
            registry["LLaDA-8B-Instruct"]     = LLaDAGenerator(target_device="cuda:0")
        if not skip_ar:
            registry["Llama-3.2-1B-Instruct"] = LlamaGenerator(target_device="cuda:1")
    else:
        registry = {}
        if not skip_diffusion:
            registry["LLaDA-8B-Instruct"]     = LLaDAGenerator()
        if not skip_ar:
            registry["Llama-3.2-1B-Instruct"] = LlamaGenerator()

    if skip_ar:
        print("[server] DO_NOT_LOAD_AR=true — skipping Llama", flush=True)
    if skip_diffusion:
        print("[server] DO_NOT_LOAD_DIFFUSION=true — skipping LLaDA", flush=True)

    loop = asyncio.get_running_loop()
    for name, gen in registry.items():
        print(f"[server] loading {name}…", flush=True)
        await loop.run_in_executor(None, gen.load)
        print(f"[server] {name} ready ✓", flush=True)
        _GENERATORS[name] = gen

    yield
    # nothing to clean up on shutdown


app = FastAPI(title="AI Timescales Inference Server", lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    tts: bool = True


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


# ── TTS schema + helpers ────────────────────────────────────────────────────
class TTSRequest(BaseModel):
    text: str
    speed: float = 1.2
    voice: str = "af_heart"


def _synth_wav(text: str, speed: float, voice: str) -> bytes | None:
    """Synthesise text → WAV bytes using Kokoro (thread-safe, cached)."""
    text = text.strip()
    if not text or _tts_pipeline is None:
        return None
    key = (text, speed, voice)
    if key in _tts_cache:
        return _tts_cache[key]
    with _tts_lock:
        chunks: list[np.ndarray] = []
        for _, _, audio in _tts_pipeline(text, voice=voice, speed=speed):
            chunks.append(audio)
    if not chunks:
        return None
    pcm = np.concatenate(chunks)
    pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm_int16.tobytes())
    result = buf.getvalue()
    _tts_cache[key] = result
    return result


@app.post("/tts")
async def tts(req: TTSRequest) -> Response:
    """Synthesise text → WAV audio (cached). Returns 204 if text is empty/TTS unavailable."""
    loop = asyncio.get_running_loop()
    wav = await loop.run_in_executor(None, _synth_wav, req.text, req.speed, req.voice)
    if wav is None:
        return Response(status_code=204)
    return Response(content=wav, media_type="audio/wav")


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

        _SKIP_TOKENS = {"<|endoftext|>", "<|eot_id|>", "<eos>", "<s>", "</s>", "<pad>"}

        while True:
            item = await q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                yield _sse("error", {"message": str(item)})
                return
            # item is a StepResult
            step_data: dict = {
                "step_index":     item.step_index,
                "token_ids":      item.token_ids,
                "decoded_tokens": item.decoded_tokens,
                "mask_positions": item.mask_positions,
                "newly_revealed": item.newly_revealed,
            }
            # ── synthesize TTS for newly-revealed tokens (parallel) ───────────
            if req.tts and item.newly_revealed:
                tok_texts = [
                    item.decoded_tokens[i]
                    for i in item.newly_revealed
                    if item.decoded_tokens[i].strip()
                    and item.decoded_tokens[i].strip() not in _SKIP_TOKENS
                ]
                if tok_texts:
                    wavs: list[bytes | None] = await asyncio.gather(*[
                        loop.run_in_executor(None, _synth_wav, t, 1.2, "af_heart")
                        for t in tok_texts
                    ])
                    # One b64 string per token (None → empty string, filtered on frontend)
                    step_data["audio_b64"] = [
                        _base64mod.b64encode(w).decode("ascii") if w else ""
                        for w in wavs
                    ]
            yield _sse("step", step_data)

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=False)


