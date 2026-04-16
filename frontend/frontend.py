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

import os
import json
from dataclasses import dataclass
from typing import Callable, Coroutine, List

import httpx
from nicegui import ui

BACKEND = "http://localhost:8000"
API_KEY = os.environ.get("API_KEY", "")   # same key, loaded from

def _client(**kwargs) -> httpx.AsyncClient:
    """httpx client pre-loaded with the API key header."""
    headers = {"X-API-Key": API_KEY}
    return httpx.AsyncClient(headers=headers, **kwargs)

CLR_MASK     = "#444"
CLR_NEW      = "#4ade80"   # green  – just generated / just revealed
CLR_EXISTING = "#60a5fa"   # blue   – already present


# SSE wire type
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


# TTS (fetched from backend /tts)
async def fetch_tts(text: str, speed: float = 1.2) -> bytes | None:
    """POST to backend /tts; returns WAV bytes or None if unavailable."""
    text = text.strip()
    if not text:
        return None
    try:
        async with _client(timeout=10.0) as client:
            r = await client.post(f"{BACKEND}/tts", json={"text": text, "speed": speed})
        if r.status_code == 204:
            return None
        r.raise_for_status()
        return r.content
    except Exception:
        return None


# HTML rendering
def token_to_html(text: str, colour: str) -> str:
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


def render_ar(ev: StepEvent) -> str:
    rev_set = set(ev.newly_revealed)
    return "".join(
        token_to_html(t, CLR_NEW if i in rev_set else CLR_EXISTING)
        for i, t in enumerate(ev.decoded_tokens)
    )


def render_diffusion(ev: StepEvent) -> str:
    mask_set = set(ev.mask_positions)
    rev_set  = set(ev.newly_revealed)
    return "".join(
        token_to_html("▒", CLR_MASK) if i in mask_set
        else token_to_html(t, CLR_NEW if i in rev_set else CLR_EXISTING)
        for i, t in enumerate(ev.decoded_tokens)
    )


# ── browser audio helpers (Web Audio API for accurate scheduling) ─────────
PLAY_ONE_JS = """
(async (b64) => {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const bytes = atob(b64);
    const buf = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) buf[i] = bytes.charCodeAt(i);
    const audioBuf = await ctx.decodeAudioData(buf.buffer);
    const src = ctx.createBufferSource();
    src.buffer = audioBuf;
    src.connect(ctx.destination);
    src.start(ctx.currentTime + 0.02);
})({b64});
"""

PLAY_MANY_JS = """
(async (entries) => {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const startTime = ctx.currentTime + 0.05;
    await Promise.all(entries.map(async (e) => {
        let audioBuf;
        if (e.mask) {
            // generate white noise in the browser
            const sr = ctx.sampleRate;
            const n = Math.floor(sr * 0.07);
            audioBuf = ctx.createBuffer(1, n, sr);
            const ch = audioBuf.getChannelData(0);
            for (let i = 0; i < n; i++) ch[i] = (Math.random() * 2 - 1) * 0.12;
        } else if (e.b64) {
            const bytes = atob(e.b64);
            const raw = new Uint8Array(bytes.length);
            for (let i = 0; i < bytes.length; i++) raw[i] = bytes.charCodeAt(i);
            try { audioBuf = await ctx.decodeAudioData(raw.buffer); }
            catch { return; }
        } else { return; }
        const src = ctx.createBufferSource();
        src.buffer = audioBuf;
        src.connect(ctx.destination);
        src.start(startTime);
    }));
})({entries});
"""


async def play_one(wav: bytes) -> None:
    import base64
    b64 = base64.b64encode(wav).decode()
    await ui.run_javascript(PLAY_ONE_JS.replace("{b64}", f'"{b64}"'))


async def play_many_diffusion(decoded_tokens: List[str], mask_positions: List[int]) -> None:
    """Fetch TTS for revealed tokens concurrently, then play all in sync via Web Audio API."""
    import base64, asyncio as _aio
    mask_set = set(mask_positions)

    # Fetch TTS for all non-mask tokens concurrently
    async def entry(idx: int, tok: str):
        if idx in mask_set:
            return {"mask": True}
        wav = await fetch_tts(tok)
        if wav:
            return {"mask": False, "b64": base64.b64encode(wav).decode()}
        return None  # skip silent tokens

    tasks = [entry(i, t) for i, t in enumerate(decoded_tokens)]
    results = await _aio.gather(*tasks)
    entries = [r for r in results if r is not None]
    if not entries:
        return
    await ui.run_javascript(PLAY_MANY_JS.replace("{entries}", json.dumps(entries)))


# page
@ui.page("/")
async def index():
    # Per-page state – no stale globals
    state = {
        "busy":           False,
        "online":         False,   # backend reachable?
        "ar_model":    None,    # populated by first successful check_server
        "diffusion_model":    None,
        "ar_prev_ids": [],
        "diffusion_prev_ids": [],
    }

    # header
    with ui.header().classes("items-center justify-between bg-slate-900 px-6 py-2"):
        ui.label("AI-Timescales").classes("text-xl font-bold")

    with ui.column().classes("w-full max-w-7xl mx-auto p-4 gap-3"):

        # last user message
        last_msg = ui.label("—").classes(
            "text-sm text-gray-300 bg-slate-800 rounded-lg px-4 py-2 self-start max-w-3xl"
        )

        # side-by-side model panels
        with ui.row().classes("w-full gap-4 items-start"):

            # AR panel
            with ui.column().classes("flex-1 min-w-0 gap-1"):
                ar_name_label = ui.label("—").classes(
                    "text-xs font-mono text-purple-400 font-semibold tracking-wide"
                )
                ar_scroll = ui.scroll_area().classes(
                    "w-full border border-purple-900/60 rounded-lg bg-slate-950"
                ).style("height:340px")
                with ar_scroll:
                    ar_html = ui.html("").classes("p-3 w-full leading-relaxed")

            # panel
            with ui.column().classes("flex-1 min-w-0 gap-1"):
                diffusion_name_label_text = ui.label("—").classes(
                    "text-xs font-mono text-emerald-400 font-semibold tracking-wide"
                )
                diffusion_scroll = ui.scroll_area().classes(
                    "w-full border border-emerald-900/60 rounded-lg bg-slate-950"
                ).style("height:340px")
                with diffusion_scroll:
                    diffusion_html = ui.html("").classes("p-3 w-full leading-relaxed")

        # settings
        with ui.row().classes("w-full items-end gap-4 flex-wrap"):
            gen_length  = ui.number("Gen length", value=8,  min=8,   max=256, step=8 ).classes("w-28")
            n_steps     = ui.number("Steps",      value=8,   min=1,   max=128, step=1 ).classes("w-24")
            temperature = ui.number("Temp",       value=0.0, min=0.0, max=2.0, step=0.1, format="%.1f").classes("w-24")
            block_len   = ui.number("Block len",  value=32,  min=8,   max=256, step=8 ).classes("w-28")
            tts_toggle  = ui.switch("TTS", value=True)

        ui.separator()

        # input + action buttons
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

        # generation progress
        gen_status = ui.label("").classes("text-xs text-gray-500")

    # server health + model discovery
    def update_buttons() -> None:
        """Enable/disable action buttons to match current state."""
        if state["busy"] or not state["online"]:
            ar_btn.disable()
            diffuse_btn.disable()
        else:
            if state["ar_model"]:
                ar_btn.enable()
            else:
                ar_btn.disable()
            if state["diffusion_model"]:
                diffuse_btn.enable()
            else:
                diffuse_btn.disable()

    def mark_offline(reason: str) -> None:
        state["online"] = False
        server_dot.content = (
            '<span style="color:#f87171;font-size:15px;line-height:1">●</span>'
        )
        server_label.text = f"Backend offline — {reason}"
        ar_btn.disable()
        diffuse_btn.disable()

    async def check_server() -> None:
        """Poll GET /status; update indicator + buttons. Runs on load + every 5 s."""
        try:
            async with _client(timeout=3.0) as client:
                r = await client.get(f"{BACKEND}/status")
                r.raise_for_status()
                data = r.json()

            models: list[str] = list(data.get("models", {}).keys())
            state["online"] = True

            # Populate model selection on first successful check
            ar_model = [m for m in models if "llada" not in m.lower()]
            diffusion_model = [m for m in models if "llada" in m.lower()]
            if state["ar_model"] is None and ar_model:
                state["ar_model"] = ar_model[0]
            if state["diffusion_model"] is None and diffusion_model:
                state["diffusion_model"] = ar_model[0]

            # Refresh panel name labels
            ar_name_label.text = state["ar_model"] or "No AR model loaded"
            diffusion_name_label_text.text = state["diffusion_model"] or "No diffusion model loaded"

            server_dot.content = (
                '<span style="color:#4ade80;font-size:15px;line-height:1">●</span>'
            )
            server_label.text = (
                f"Backend online — {', '.join(models)}"
                if models else
                "Backend online — no models loaded yet"
            )
            update_buttons()

        except httpx.ConnectError:
            mark_offline("cannot connect to port 8000")
        except httpx.TimeoutException:
            mark_offline("connection timed out")
        except Exception as exc:
            mark_offline(f"{type(exc).__name__}: {str(exc)[:60]}")

    await check_server()
    ui.timer(5.0, check_server)

    # SSE stream helper
    async def stream(
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
            async with _client(
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
            mark_offline("connection lost during generation")
        except Exception as exc:
            gen_status.text = f"⚠ {exc}"
        return False

    # shared generation helpers
    def begin(msg: str) -> None:
        """Shared setup: set busy, update buttons/stop, clear status."""
        state["busy"] = True
        update_buttons()
        stop_btn.text = "Stop"
        stop_btn.props("icon=stop")
        last_msg.text = msg
        gen_status.text = ""

    def end(ok: bool) -> None:
        """Shared teardown: clear busy flag, update status text + buttons."""
        if ok:
            gen_status.text = "✓ Done"
        elif gen_status.text.startswith("⟳"):
            # stream returned False but spinner still showing → silent cancel
            gen_status.text = "Stopped."
        state["busy"] = False
        update_buttons()
        stop_btn.text = "Clear"
        stop_btn.props("icon=delete")

    def params() -> tuple[bool, int, int, int, float]:
        return (
            tts_toggle.value,
            int(gen_length.value),
            int(n_steps.value),
            int(block_len.value),
            float(temperature.value),
        )

    # AUTOREGRESS button
    async def on_autoregress() -> None:
        if state["busy"] or not state["online"]:
            return
        msg = user_input.value.strip()
        if not msg:
            return
        user_input.value = ""

        begin(msg)
        ar_html.content = ""
        do_tts, gl, st, bl, tmp = params()
        ok = True

        if state["ar_model"]:
            gen_status.text = f"⟳ {state['ar_model']}…"

            async def ar_step(ev: StepEvent) -> None:
                wav = None
                if do_tts and ev.newly_revealed:
                    tok = ev.decoded_tokens[ev.newly_revealed[0]]
                    if tok.strip():
                        wav = await fetch_tts(tok)
                if wav:
                    await play_one(wav)
                ar_html.content = render_ar(ev)
                ar_scroll.scroll_to(percent=1.0)

            async def ar_done(data: dict) -> None:
                state["ar_prev_ids"] = data.get("final_token_ids", [])

            ok = await stream(
                payload={
                    "model":          state["ar_model"],
                    "messages":       [{"role": "user", "content": msg}],
                    "prev_token_ids": state["ar_prev_ids"],
                    "gen_length": gl, "steps": st,
                    "block_length": bl, "temperature": tmp,
                },
                on_step=ar_step,
                on_done=ar_done,
            )

        end(ok)

    # DIFFUSE button
    async def on_diffuse() -> None:
        if state["busy"] or not state["online"]:
            return
        msg = user_input.value.strip()
        if not msg:
            return
        user_input.value = ""

        begin(msg)
        diffusion_html.content = ""
        do_tts, gl, st, bl, tmp = params()
        ok = True

        if state["diffusion_model"]:
            gen_status.text = f"⟳ {state['diffusion_model']}…"

            async def diffusion_step(ev: StepEvent) -> None:
                if do_tts:
                    await play_many_diffusion(ev.decoded_tokens, ev.mask_positions)  # fetch + start
                hdr = (
                    f"<span style='color:#555;font-size:11px;font-family:monospace'>"
                    f"step {ev.step_index + 1}</span><br>"
                )
                diffusion_html.content = hdr + render_diffusion(ev)  # reveal after audio starts
                diffusion_scroll.scroll_to(percent=1.0)

            async def diffusion_done(data: dict) -> None:
                state["diffusion_prev_ids"] = data.get("final_token_ids", [])

            ok = await stream(
                payload={
                    "model":          state["diffusion_model"],
                    "messages":       [{"role": "user", "content": msg}],
                    "prev_token_ids": state["diffusion_prev_ids"],
                    "gen_length": gl, "steps": st,
                    "block_length": bl, "temperature": tmp,
                },
                on_step=diffusion_step,
                on_done=diffusion_done,
            )

        end(ok)

    # ── Enter key → run both sequentially (LLaMA first, then LLaDA) ──────
    async def on_both() -> None:
        if state["busy"] or not state["online"]:
            return
        msg = user_input.value.strip()
        if not msg:
            return
        user_input.value = ""

        begin(msg)
        ar_html.content = ""
        diffusion_html.content = ""
        do_tts, gl, st, bl, tmp = params()
        ok = True

        if state["ar_model"] and ok:
            gen_status.text = f"⟳ {state['ar_model']}…"

            async def both_ar_step(ev: StepEvent) -> None:
                wav = None
                if do_tts and ev.newly_revealed:
                    tok = ev.decoded_tokens[ev.newly_revealed[0]]
                    if tok.strip():
                        wav = await fetch_tts(tok)
                if wav:
                    await play_one(wav)
                ar_html.content = render_ar(ev)
                ar_scroll.scroll_to(percent=1.0)

            async def both_ar_done(data: dict) -> None:
                state["ar_prev_ids"] = data.get("final_token_ids", [])

            ok = await stream(
                payload={
                    "model": state["ar_model"],
                    "messages": [{"role": "user", "content": msg}],
                    "prev_token_ids": state["ar_prev_ids"],
                    "gen_length": gl, "steps": st,
                    "block_length": bl, "temperature": tmp,
                },
                on_step=both_ar_step,
                on_done=both_ar_done,
            )

        if state["diffusion_model"] and ok:
            gen_status.text = f"⟳ {state['diffusion_model']}…"

            async def both_diffusion_step(ev: StepEvent) -> None:
                hdr = (
                    f"<span style='color:#555;font-size:11px;font-family:monospace'>"
                    f"step {ev.step_index + 1}</span><br>"
                )
                diffusion_html.content = hdr + render_diffusion(ev)
                diffusion_scroll.scroll_to(percent=1.0)
                if do_tts:
                    await play_many_diffusion(ev.decoded_tokens, ev.mask_positions)

            async def both_diffusion_done(data: dict) -> None:
                state["diffusion_prev_ids"] = data.get("final_token_ids", [])

            ok = await stream(
                payload={
                    "model":          state["diffusion_model"],
                    "messages":       [{"role": "user", "content": msg}],
                    "prev_token_ids": state["diffusion_prev_ids"],
                    "gen_length": gl, "steps": st,
                    "block_length": bl, "temperature": tmp,
                },
                on_step=both_diffusion_step,
                on_done=both_diffusion_done,
            )

        end(ok)

    # stop / clear
    async def on_stop_or_clear() -> None:
        if state["busy"]:
            try:
                async with _client(timeout=5.0) as client:
                    await client.post(f"{BACKEND}/stop")
            except Exception:
                pass
        else:
            ar_html.content = ""
            diffusion_html.content = ""
            last_msg.text = "—"
            gen_status.text = ""
            state["ar_prev_ids"] = []
            state["diffusion_prev_ids"] = []

    # bind events
    ar_btn.on_click(on_autoregress)
    diffuse_btn.on_click(on_diffuse)
    user_input.on("keydown.enter", on_both)
    stop_btn.on_click(on_stop_or_clear)


# launch
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="AI-Timescales", host="0.0.0.0", port=8080, reload=False)
