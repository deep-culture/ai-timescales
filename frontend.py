"""
AI-Timescales – dual-model comparison UI.

Layout
  • Small "last message" bar at the top
  • LLaMA (AR) output  |  LLaDA (diffusion) output   — side by side
  • "AUTOREGRESS" button  → runs only the LLaMA AR panel
  • "DIFFUSE" button      → runs only the LLaDA diffusion panel
  • Enter key             → runs both sequentially (LLaMA first, then LLaDA)
  • LLaMA vocalises each new token immediately as it is generated
  • LLaDA vocalises all tokens simultaneously per denoising step;
    MASK positions play a short white-noise burst

Start:
    uvicorn backend:app --port 8000 --workers 1
    python frontend.py
Then open http://localhost:8080
"""
from __future__ import annotations

import io
import base64
import json
import threading
import wave
from dataclasses import dataclass
from typing import Callable, Coroutine, List

import httpx
import numpy as np
from nicegui import ui, run

BACKEND = "http://localhost:8000"

# ── token display colours ─────────────────────────────────────────────────
CLR_MASK     = "#444"
CLR_NEW      = "#4ade80"   # green  – just generated / just revealed
CLR_EXISTING = "#60a5fa"   # blue   – already present


# ── SSE wire type ─────────────────────────────────────────────────────────
@dataclass
class StepEvent:
    step_index: int
    token_ids: List[int]
    decoded_tokens: List[str]
    mask_positions: List[int]
    newly_revealed: List[int]

    @staticmethod
    def from_dict(d: dict) -> "StepEvent":
        return StepEvent(
            step_index=d["step_index"],
            token_ids=d["token_ids"],
            decoded_tokens=d["decoded_tokens"],
            mask_positions=d["mask_positions"],
            newly_revealed=d["newly_revealed"],
        )


# ── TTS ───────────────────────────────────────────────────────────────────
_kokoro_pipeline = None
_tts_lock = threading.Lock()
_tts_cache: dict[str, bytes] = {}   # token text → WAV bytes (avoids re-synth)


def _get_tts():
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        from kokoro import KPipeline
        _kokoro_pipeline = KPipeline(lang_code="a")
    return _kokoro_pipeline


def _synth(text: str, speed: float = 1.2) -> bytes | None:
    """Synthesise one text string → WAV bytes (cached per unique string)."""
    text = text.strip()
    if not text:
        return None
    if text in _tts_cache:
        return _tts_cache[text]
    with _tts_lock:
        pipeline = _get_tts()
    chunks: list[np.ndarray] = []
    for _, _, audio in pipeline(text, voice="af_heart", speed=speed):
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
    _tts_cache[text] = result
    return result


def _white_noise_wav(duration_s: float = 0.07, sample_rate: int = 24000) -> bytes:
    """Short white-noise burst representing a MASK token."""
    n = int(duration_s * sample_rate)
    noise = np.random.uniform(-0.12, 0.12, n).astype(np.float32)
    pcm_int16 = (noise * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


def _make_llada_audio(decoded_tokens: List[str], mask_positions: List[int]) -> list[bytes]:
    """
    One WAV per token position:
      mask     → white noise  (instant, no TTS)
      revealed → _synth()     (cached after first call)
    """
    mask_set = set(mask_positions)
    results: list[bytes] = []
    for idx, tok in enumerate(decoded_tokens):
        wav = _white_noise_wav() if idx in mask_set else _synth(tok)
        if wav:
            results.append(wav)
    return results


# ── HTML rendering ────────────────────────────────────────────────────────
def _tok_html(text: str, colour: str) -> str:
    safe = (
        text.replace("&", "&amp;").replace("<", "&lt;")
        .replace(">", "&gt;").replace(" ", "&nbsp;")
    )
    if not safe:
        safe = "&nbsp;"
    return (
        f'<span style="color:{colour};font-family:monospace;padding:1px 3px;margin:1px;'
        f'border-radius:3px;background:rgba(255,255,255,0.07);display:inline-block">'
        f"{safe}</span>"
    )


def _render_llama(ev: StepEvent) -> str:
    rev_set = set(ev.newly_revealed)
    return "".join(
        _tok_html(t, CLR_NEW if i in rev_set else CLR_EXISTING)
        for i, t in enumerate(ev.decoded_tokens)
    )


def _render_llada(ev: StepEvent) -> str:
    mask_set = set(ev.mask_positions)
    rev_set  = set(ev.newly_revealed)
    return "".join(
        _tok_html("▒", CLR_MASK) if i in mask_set
        else _tok_html(t, CLR_NEW if i in rev_set else CLR_EXISTING)
        for i, t in enumerate(ev.decoded_tokens)
    )


# ── browser audio helpers ─────────────────────────────────────────────────
async def _play_one(wav: bytes) -> None:
    b64 = base64.b64encode(wav).decode()
    await ui.run_javascript(f"new Audio('data:audio/wav;base64,{b64}').play();")


async def _play_many(wavs: list[bytes]) -> None:
    if not wavs:
        return
    b64s = json.dumps([base64.b64encode(w).decode() for w in wavs])
    await ui.run_javascript(
        f"(function(){{const ws={b64s};"
        f"ws.map(b=>new Audio('data:audio/wav;base64,'+b)).forEach(a=>a.play());}})();"
    )


# ── page ──────────────────────────────────────────────────────────────────
@ui.page("/")
async def index():
    # Per-page state – no stale globals
    state = {
        "busy":           False,
        "online":         False,   # backend reachable?
        "llama_model":    None,    # populated by first successful _check_server
        "llada_model":    None,
        "llama_prev_ids": [],
        "llada_prev_ids": [],
    }

    # ── header ───────────────────────────────────────────────────────────
    with ui.header().classes("items-center justify-between bg-slate-900 px-6 py-2"):
        ui.label("AI-Timescales").classes("text-xl font-bold")

    with ui.column().classes("w-full max-w-7xl mx-auto p-4 gap-3"):

        # ── last user message ─────────────────────────────────────────────
        last_msg = ui.label("—").classes(
            "text-sm text-gray-300 bg-slate-800 rounded-lg px-4 py-2 self-start max-w-3xl"
        )

        # ── side-by-side model panels ─────────────────────────────────────
        with ui.row().classes("w-full gap-4 items-start"):

            # LLaMA (AR) panel ────────────────────────────────────────────
            with ui.column().classes("flex-1 min-w-0 gap-1"):
                llama_name_label = ui.label("—").classes(
                    "text-xs font-mono text-purple-400 font-semibold tracking-wide"
                )
                llama_scroll = ui.scroll_area().classes(
                    "w-full border border-purple-900/60 rounded-lg bg-slate-950"
                ).style("height:340px")
                with llama_scroll:
                    llama_html = ui.html("").classes("p-3 w-full leading-relaxed")

            # LLaDA (diffusion) panel ─────────────────────────────────────
            with ui.column().classes("flex-1 min-w-0 gap-1"):
                llada_name_label = ui.label("—").classes(
                    "text-xs font-mono text-emerald-400 font-semibold tracking-wide"
                )
                llada_scroll = ui.scroll_area().classes(
                    "w-full border border-emerald-900/60 rounded-lg bg-slate-950"
                ).style("height:340px")
                with llada_scroll:
                    llada_html = ui.html("").classes("p-3 w-full leading-relaxed")

        # ── settings ──────────────────────────────────────────────────────
        with ui.row().classes("w-full items-end gap-4 flex-wrap"):
            gen_length  = ui.number("Gen length", value=64,  min=8,   max=256, step=8 ).classes("w-28")
            n_steps     = ui.number("Steps",      value=8,   min=1,   max=128, step=1 ).classes("w-24")
            temperature = ui.number("Temp",       value=0.0, min=0.0, max=2.0, step=0.1, format="%.1f").classes("w-24")
            block_len   = ui.number("Block len",  value=32,  min=8,   max=256, step=8 ).classes("w-28")
            tts_toggle  = ui.switch("TTS", value=True)

        ui.separator()

        # ── input + action buttons ────────────────────────────────────────
        with ui.row().classes("w-full items-center gap-2"):
            user_input = ui.input(placeholder="Type your message…").props(
                "outlined dense"
            ).classes("flex-grow")
            # AR button – purple, matches LLaMA panel
            ar_btn = ui.button("Autoregress", icon="fast_forward").props(
                'color="purple"'
            )
            ar_btn.disable()
            # Diffusion button – teal, matches LLaDA panel
            diffuse_btn = ui.button("Diffuse", icon="blur_on").props(
                'color="teal"'
            )
            diffuse_btn.disable()
            # Stop / Clear toggle
            stop_btn = ui.button("Clear", icon="delete", color="negative")

        # ── server status row (always visible) ────────────────────────────
        with ui.row().classes("w-full items-center gap-2 mt-1"):
            server_dot   = ui.html(
                '<span style="color:#888;font-size:15px;line-height:1">●</span>'
            )
            server_label = ui.label("Connecting to backend…").classes(
                "text-xs text-gray-400"
            )

        # ── generation progress ───────────────────────────────────────────
        gen_status = ui.label("").classes("text-xs text-gray-500")

    # ── server health + model discovery ───────────────────────────────────
    def _update_buttons() -> None:
        """Enable/disable action buttons to match current state."""
        if state["busy"] or not state["online"]:
            ar_btn.disable()
            diffuse_btn.disable()
        else:
            if state["llama_model"]:
                ar_btn.enable()
            else:
                ar_btn.disable()
            if state["llada_model"]:
                diffuse_btn.enable()
            else:
                diffuse_btn.disable()

    def _mark_offline(reason: str) -> None:
        state["online"] = False
        server_dot.content = (
            '<span style="color:#f87171;font-size:15px;line-height:1">●</span>'
        )
        server_label.text = f"Backend offline — {reason}"
        ar_btn.disable()
        diffuse_btn.disable()

    async def _check_server() -> None:
        """Poll GET /status; update indicator + buttons. Runs on load + every 5 s."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{BACKEND}/status")
                r.raise_for_status()
                data = r.json()

            models: list[str] = list(data.get("models", {}).keys())
            state["online"] = True

            # Populate model selection on first successful check
            llama_m = [m for m in models if "llada" not in m.lower()]
            llada_m = [m for m in models if "llada"     in m.lower()]
            if state["llama_model"] is None and llama_m:
                state["llama_model"] = llama_m[0]
            if state["llada_model"] is None and llada_m:
                state["llada_model"] = llada_m[0]

            # Refresh panel name labels
            llama_name_label.text = state["llama_model"] or "No AR model loaded"
            llada_name_label.text = state["llada_model"] or "No diffusion model loaded"

            server_dot.content = (
                '<span style="color:#4ade80;font-size:15px;line-height:1">●</span>'
            )
            server_label.text = (
                f"Backend online — {', '.join(models)}"
                if models else
                "Backend online — no models loaded yet"
            )
            _update_buttons()

        except httpx.ConnectError:
            _mark_offline("cannot connect to port 8000")
        except httpx.TimeoutException:
            _mark_offline("connection timed out")
        except Exception as exc:
            _mark_offline(f"{type(exc).__name__}: {str(exc)[:60]}")

    await _check_server()
    ui.timer(5.0, _check_server)

    # ── SSE stream helper ─────────────────────────────────────────────────
    async def _stream(
        payload: dict,
        on_step: Callable[[StepEvent], Coroutine],
        on_done: Callable[[dict], Coroutine],
    ) -> bool:
        """
        POST /generate and consume SSE events.
        Returns True on clean completion, False on cancel / error / unreachable.
        Always sets gen_status.text on failure.
        """
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, read=None)
            ) as client:
                async with client.stream(
                    "POST", f"{BACKEND}/generate", json=payload
                ) as resp:
                    if resp.status_code != 200:
                        gen_status.text = f"⚠ Server error {resp.status_code}"
                        return False
                    event_type: str | None = None
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                        elif line.startswith("data:"):
                            data = json.loads(line[5:].strip())
                            if event_type == "step":
                                await on_step(StepEvent.from_dict(data))
                            elif event_type == "done":
                                await on_done(data)
                                return True
                            elif event_type == "cancelled":
                                gen_status.text = "Generation stopped."
                                return False
                            elif event_type == "error":
                                gen_status.text = f"⚠ {data.get('message', 'unknown error')}"
                                return False
        except httpx.ConnectError:
            gen_status.text = "⚠ Lost connection to backend."
            _mark_offline("connection lost during generation")
        except Exception as exc:
            gen_status.text = f"⚠ {exc}"
        return False

    # ── shared generation helpers ─────────────────────────────────────────
    def _begin(msg: str) -> None:
        """Shared setup: set busy, update buttons/stop, clear status."""
        state["busy"] = True
        _update_buttons()
        stop_btn.text = "Stop"
        stop_btn.props("icon=stop")
        last_msg.text = msg
        gen_status.text = ""

    def _end(ok: bool) -> None:
        """Shared teardown: clear busy flag, update status text + buttons."""
        if ok:
            gen_status.text = "✓ Done"
        elif gen_status.text.startswith("⟳"):
            # _stream returned False but spinner still showing → silent cancel
            gen_status.text = "Stopped."
        state["busy"] = False
        _update_buttons()
        stop_btn.text = "Clear"
        stop_btn.props("icon=delete")

    def _params() -> tuple[bool, int, int, int, float]:
        return (
            tts_toggle.value,
            int(gen_length.value),
            int(n_steps.value),
            int(block_len.value),
            float(temperature.value),
        )

    # ── AUTOREGRESS button ────────────────────────────────────────────────
    async def on_autoregress() -> None:
        if state["busy"] or not state["online"]:
            return
        msg = user_input.value.strip()
        if not msg:
            return
        user_input.value = ""

        _begin(msg)
        llama_html.content = ""
        do_tts, gl, st, bl, tmp = _params()
        ok = True

        if state["llama_model"]:
            gen_status.text = f"⟳ {state['llama_model']}…"

            async def llama_step(ev: StepEvent) -> None:
                llama_html.content = _render_llama(ev)
                llama_scroll.scroll_to(percent=1.0)
                if do_tts and ev.newly_revealed:
                    tok = ev.decoded_tokens[ev.newly_revealed[0]]
                    if tok.strip():
                        wav = await run.io_bound(_synth, tok)
                        if wav:
                            await _play_one(wav)

            async def llama_done(data: dict) -> None:
                state["llama_prev_ids"] = data.get("final_token_ids", [])

            ok = await _stream(
                payload={
                    "model":          state["llama_model"],
                    "messages":       [{"role": "user", "content": msg}],
                    "prev_token_ids": state["llama_prev_ids"],
                    "gen_length": gl, "steps": st,
                    "block_length": bl, "temperature": tmp,
                },
                on_step=llama_step,
                on_done=llama_done,
            )

        _end(ok)

    # ── DIFFUSE button ────────────────────────────────────────────────────
    async def on_diffuse() -> None:
        if state["busy"] or not state["online"]:
            return
        msg = user_input.value.strip()
        if not msg:
            return
        user_input.value = ""

        _begin(msg)
        llada_html.content = ""
        do_tts, gl, st, bl, tmp = _params()
        ok = True

        if state["llada_model"]:
            gen_status.text = f"⟳ {state['llada_model']}…"

            async def llada_step(ev: StepEvent) -> None:
                hdr = (
                    f"<span style='color:#555;font-size:11px;font-family:monospace'>"
                    f"step {ev.step_index + 1}</span><br>"
                )
                llada_html.content = hdr + _render_llada(ev)
                llada_scroll.scroll_to(percent=1.0)
                if do_tts:
                    wavs = await run.io_bound(
                        _make_llada_audio, ev.decoded_tokens, ev.mask_positions
                    )
                    if wavs:
                        await _play_many(wavs)

            async def llada_done(data: dict) -> None:
                state["llada_prev_ids"] = data.get("final_token_ids", [])

            ok = await _stream(
                payload={
                    "model":          state["llada_model"],
                    "messages":       [{"role": "user", "content": msg}],
                    "prev_token_ids": state["llada_prev_ids"],
                    "gen_length": gl, "steps": st,
                    "block_length": bl, "temperature": tmp,
                },
                on_step=llada_step,
                on_done=llada_done,
            )

        _end(ok)

    # ── Enter key → run both sequentially (LLaMA first, then LLaDA) ──────
    async def on_both() -> None:
        if state["busy"] or not state["online"]:
            return
        msg = user_input.value.strip()
        if not msg:
            return
        user_input.value = ""

        _begin(msg)
        llama_html.content = ""
        llada_html.content = ""
        do_tts, gl, st, bl, tmp = _params()
        ok = True

        if state["llama_model"] and ok:
            gen_status.text = f"⟳ {state['llama_model']}…"

            async def both_llama_step(ev: StepEvent) -> None:
                llama_html.content = _render_llama(ev)
                llama_scroll.scroll_to(percent=1.0)
                if do_tts and ev.newly_revealed:
                    tok = ev.decoded_tokens[ev.newly_revealed[0]]
                    if tok.strip():
                        wav = await run.io_bound(_synth, tok)
                        if wav:
                            await _play_one(wav)

            async def both_llama_done(data: dict) -> None:
                state["llama_prev_ids"] = data.get("final_token_ids", [])

            ok = await _stream(
                payload={
                    "model":          state["llama_model"],
                    "messages":       [{"role": "user", "content": msg}],
                    "prev_token_ids": state["llama_prev_ids"],
                    "gen_length": gl, "steps": st,
                    "block_length": bl, "temperature": tmp,
                },
                on_step=both_llama_step,
                on_done=both_llama_done,
            )

        if state["llada_model"] and ok:
            gen_status.text = f"⟳ {state['llada_model']}…"

            async def both_llada_step(ev: StepEvent) -> None:
                hdr = (
                    f"<span style='color:#555;font-size:11px;font-family:monospace'>"
                    f"step {ev.step_index + 1}</span><br>"
                )
                llada_html.content = hdr + _render_llada(ev)
                llada_scroll.scroll_to(percent=1.0)
                if do_tts:
                    wavs = await run.io_bound(
                        _make_llada_audio, ev.decoded_tokens, ev.mask_positions
                    )
                    if wavs:
                        await _play_many(wavs)

            async def both_llada_done(data: dict) -> None:
                state["llada_prev_ids"] = data.get("final_token_ids", [])

            ok = await _stream(
                payload={
                    "model":          state["llada_model"],
                    "messages":       [{"role": "user", "content": msg}],
                    "prev_token_ids": state["llada_prev_ids"],
                    "gen_length": gl, "steps": st,
                    "block_length": bl, "temperature": tmp,
                },
                on_step=both_llada_step,
                on_done=both_llada_done,
            )

        _end(ok)

    # ── stop / clear ──────────────────────────────────────────────────────
    async def on_stop_or_clear() -> None:
        if state["busy"]:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    await client.post(f"{BACKEND}/stop")
            except Exception:
                pass
        else:
            llama_html.content = ""
            llada_html.content = ""
            last_msg.text = "—"
            gen_status.text = ""
            state["llama_prev_ids"] = []
            state["llada_prev_ids"] = []

    # ── bind events ───────────────────────────────────────────────────────
    ar_btn.on_click(on_autoregress)
    diffuse_btn.on_click(on_diffuse)
    user_input.on("keydown.enter", on_both)
    stop_btn.on_click(on_stop_or_clear)


# ── launch ────────────────────────────────────────────────────────────────
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="AI-Timescales", port=8080, reload=False)
