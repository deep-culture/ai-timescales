"""
AI-Timescales – NiceGUI chat app with per-step token visualisation & TTS.

Run:  python app.py
Then open http://localhost:8080
"""
from __future__ import annotations

import asyncio
import io
import base64
import uuid
from typing import Dict

import numpy as np
import torch
from nicegui import ui, app, run

from inference import LLaDAGenerator, StepResult

# ── TTS pipeline (lazy-loaded) ──────────────────────────────────────────
_kokoro_pipeline = None


def _get_tts():
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        from kokoro import KPipeline
        _kokoro_pipeline = KPipeline(lang_code="a")
    return _kokoro_pipeline


def tts_to_wav_bytes(text: str, speed: float = 1.2) -> bytes | None:
    """Run Kokoro TTS and return raw WAV bytes (24 kHz, mono, float32)."""
    import wave

    text = text.strip()
    if not text:
        return None

    pipeline = _get_tts()
    chunks: list[np.ndarray] = []
    for _, _, audio_array in pipeline(text, voice="af_heart", speed=speed):
        chunks.append(audio_array)

    if not chunks:
        return None

    pcm = np.concatenate(chunks)
    # Convert float32 [-1,1] → int16 for WAV
    pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(24000)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


# ── Generators registry ─────────────────────────────────────────────────
GENERATORS: Dict[str, LLaDAGenerator] = {
    "LLaDA-8B-Instruct": LLaDAGenerator(),
}

MASK_TOKEN_DISPLAY = "[MASK]"

# ── Colour helpers ───────────────────────────────────────────────────────
CLR_MASK = "#555"
CLR_NEW = "#4ade80"      # green
CLR_EXISTING = "#60a5fa" # blue


def _token_html(text: str, colour: str) -> str:
    """Wrap a token string in a coloured <span>."""
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace(" ", "&nbsp;")
    )
    if not safe:
        safe = "&nbsp;"
    return (
        f'<span style="color:{colour};font-family:monospace;'
        f'padding:1px 3px;margin:1px;border-radius:3px;'
        f'background:rgba(255,255,255,0.07);display:inline-block">'
        f"{safe}</span>"
    )


def step_to_html(result: StepResult) -> str:
    """Render a StepResult as an HTML string with colour-coded tokens."""
    parts: list[str] = []
    for idx, tok_text in enumerate(result.decoded_tokens):
        if idx in result.mask_positions:
            parts.append(_token_html(MASK_TOKEN_DISPLAY, CLR_MASK))
        elif idx in result.newly_revealed:
            parts.append(_token_html(tok_text, CLR_NEW))
        else:
            parts.append(_token_html(tok_text, CLR_EXISTING))
    return "".join(parts)


# ── NiceGUI page ─────────────────────────────────────────────────────────
@ui.page("/")
async def index():
    # ── per-client state ─────────────────────────────────────────────
    state = {
        "generator": GENERATORS["LLaDA-8B-Instruct"],
        "prompt_ids": None,          # running prompt tensor
        "conversation_num": 0,
        "busy": False,
    }

    # ── header ───────────────────────────────────────────────────────
    with ui.header().classes("items-center justify-between bg-slate-900"):
        ui.label("AI-Timescales").classes("text-xl font-bold")

    # ── main layout ──────────────────────────────────────────────────
    with ui.column().classes("w-full max-w-5xl mx-auto p-4 gap-4"):

        # -- model selector + settings row --
        with ui.row().classes("w-full items-end gap-4"):
            model_select = ui.select(
                list(GENERATORS.keys()),
                value="LLaDA-8B-Instruct",
                label="Model",
            ).classes("w-60")

            gen_length = ui.number("Gen length", value=64, min=8, max=256, step=8).classes("w-28")
            n_steps = ui.number("Steps", value=8, min=1, max=128, step=1).classes("w-24")
            temperature = ui.number("Temp", value=0.0, min=0.0, max=2.0, step=0.1, format="%.1f").classes("w-24")
            block_len = ui.number("Block len", value=32, min=8, max=256, step=8).classes("w-28")
            tts_toggle = ui.switch("TTS", value=True)

        ui.separator()

        # -- chat history --
        chat_scroll = ui.scroll_area().classes("w-full border rounded-lg bg-slate-950").style("height:300px")
        chat_column = ui.column().classes("w-full gap-2 p-2")
        # Move chat_column inside scroll area
        chat_column.move(chat_scroll)

        # -- step-by-step visualisation card --
        ui.label("Denoising steps").classes("text-sm text-gray-400 mt-2")
        step_scroll = ui.scroll_area().classes("w-full border rounded-lg bg-slate-950").style("height:260px")
        step_column = ui.column().classes("w-full gap-1 p-2")
        step_column.move(step_scroll)

        # -- audio container (hidden) — we inject <audio> elements here --
        audio_container = ui.element("div").classes("hidden")

        # -- input row --
        with ui.row().classes("w-full items-center gap-2"):
            user_input = ui.input(placeholder="Type your message…").props(
                "outlined dense"
            ).classes("flex-grow")
            send_btn = ui.button("Send", icon="send")
            clear_btn = ui.button("Clear", icon="delete", color="negative")

        # -- status bar --
        status = ui.label("Ready").classes("text-xs text-gray-500")

    # ── helpers ───────────────────────────────────────────────────────

    def _add_bubble(role: str, text: str):
        """Append a chat bubble to the history."""
        is_user = role == "user"
        align = "items-end" if is_user else "items-start"
        bg = "bg-blue-900" if is_user else "bg-slate-800"
        with chat_column:
            with ui.column().classes(f"w-full {align}"):
                ui.label(text).classes(
                    f"{bg} text-white rounded-lg px-3 py-2 max-w-lg whitespace-pre-wrap text-sm"
                )
        chat_scroll.scroll_to(percent=1.0)

    async def _ensure_loaded():
        gen = state["generator"]
        if not gen.is_loaded():
            status.text = "Loading model (this may take a while)…"
            await run.io_bound(gen.load)   # io_bound = thread pool = shared memory
            status.text = "Model loaded ✓"

    # ── send handler ─────────────────────────────────────────────────
    async def on_send():
        if state["busy"]:
            return
        msg = user_input.value.strip()
        if not msg:
            return

        state["busy"] = True
        send_btn.disable()
        user_input.value = ""
        _add_bubble("user", msg)
        status.text = "Loading model…"

        await _ensure_loaded()

        gen: LLaDAGenerator = state["generator"]

        # ── tokenise ────────────────────────────────────
        messages = [{"role": "user", "content": msg}]
        input_ids = gen.encode_chat(messages)

        if state["conversation_num"] == 0:
            prompt = input_ids
        else:
            prompt = torch.cat([state["prompt_ids"], input_ids[:, 1:]], dim=1)

        prompt_len = prompt.shape[1]

        # ── clear step viewer ───────────────────────────
        step_column.clear()

        status.text = "Generating…"

        # ── run generation in a thread, push UI updates ─
        gl = int(gen_length.value)
        st = int(n_steps.value)
        bl = int(block_len.value)
        temp = float(temperature.value)
        do_tts = tts_toggle.value

        # We collect results from the background thread via a queue
        result_queue: asyncio.Queue[StepResult | None] = asyncio.Queue()

        def _run_gen():
            """Runs in a worker thread."""
            for step_result in gen.generate_steps(
                prompt,
                gen_length=gl,
                steps=st,
                block_length=bl,
                temperature=temp,
            ):
                asyncio.run_coroutine_threadsafe(
                    result_queue.put(step_result),
                    loop,
                )
            # Sentinel
            asyncio.run_coroutine_threadsafe(result_queue.put(None), loop)

        loop = asyncio.get_event_loop()
        gen_task = loop.run_in_executor(None, _run_gen)

        # ── consume results and update UI ───────────────
        while True:
            step_result = await result_queue.get()
            if step_result is None:
                break

            html = step_to_html(step_result)
            header = f"<b style='color:#aaa;font-size:12px'>Step {step_result.step_index + 1}</b>"
            with step_column:
                ui.html(f"{header}<br>{html}").classes("w-full")
            step_scroll.scroll_to(percent=1.0)

            # ── TTS for this step (non-blocking) ────────
            if do_tts:
                # Decode only the newly-revealed tokens for speech
                revealed_text = " ".join(
                    step_result.decoded_tokens[i]
                    for i in step_result.newly_revealed
                )
                if revealed_text.strip():
                    wav = await run.io_bound(tts_to_wav_bytes, revealed_text)
                    if wav:
                        b64 = base64.b64encode(wav).decode()
                        uid = uuid.uuid4().hex[:8]
                        await ui.run_javascript(
                            f"new Audio('data:audio/wav;base64,{b64}').play();"
                        )

        await gen_task  # ensure thread finished

        # ── final answer ────────────────────────────────
        answer = gen.get_final_text()
        _add_bubble("assistant", answer)
        status.text = "Ready"

        # ── update running prompt for multi-turn ────────
        if gen._final_x is not None:
            state["prompt_ids"] = gen.strip_eos(gen._final_x)
        state["conversation_num"] += 1
        state["busy"] = False
        send_btn.enable()

    # ── clear handler ────────────────────────────────────────────────
    def on_clear():
        chat_column.clear()
        step_column.clear()
        state["prompt_ids"] = None
        state["conversation_num"] = 0
        status.text = "Conversation cleared"

    # ── bind events ──────────────────────────────────────────────────
    send_btn.on_click(on_send)
    user_input.on("keydown.enter", on_send)
    clear_btn.on_click(on_clear)

    def on_model_change(e):
        name = e.value
        state["generator"] = GENERATORS[name]
        state["prompt_ids"] = None
        state["conversation_num"] = 0
        chat_column.clear()
        step_column.clear()
        status.text = f"Switched to {name}"

    model_select.on_value_change(on_model_change)


# ── launch ───────────────────────────────────────────────────────────────
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="AI-Timescales", port=8080, reload=False)

