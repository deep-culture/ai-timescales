<template>
  <header>
    <span class="title">AI-Timescales</span>
    <span :class="['dot', online ? 'on' : 'off']">●</span>
    <span class="status-text">{{ statusText }}</span>
  </header>

  <main>
    <div class="last-msg">{{ lastMsg }}</div>

    <!-- heartbeat graph -->
    <HeartbeatGraph :points="heartbeat" />

    <!-- single merged output box -->
    <div class="output" ref="outputEl">
      <span v-for="(tok, i) in tokens" :key="i" :class="['tok', tok.cls]">{{ tok.text }}</span>
    </div>

    <!-- params -->
    <div class="params">
      <label>Gen length <input type="number" v-model.number="genLength" min="8" max="256" step="8" @change="snapConstraints()" /></label>
      <label>Steps <input type="number" v-model.number="steps" min="1" max="256" step="1" @change="snapConstraints()" /></label>
      <label>Temp <input type="number" v-model.number="temperature" min="0" max="2" step="0.1" /></label>
      <label>Block len <input type="number" v-model.number="blockLength" min="8" max="256" step="8" @change="snapConstraints()" /></label>
      <span class="note">{{ constraintNote }}</span>
    </div>

    <!-- input + buttons -->
    <div class="controls">
      <input class="msg-input" v-model="userInput" placeholder="Type your message…" @keydown.enter="onEnter" />
      <button class="btn ar" :disabled="busy || !online || !arModel" @click="runAR">Autoregress</button>
      <button class="btn diff" :disabled="busy || !online || !diffusionModel" @click="runDiffuse">Diffuse</button>
      <button class="btn stop" @click="stopOrClear">{{ busy ? 'Stop' : 'Clear' }}</button>
      <button :class="['btn', 'tts', ttsEnabled ? 'tts-on' : 'tts-off']" @click="ttsEnabled = !ttsEnabled" title="Toggle TTS">{{ ttsEnabled ? '🔊' : '🔇' }}</button>
    </div>

    <div class="gen-status">{{ genStatus }}</div>

    <!-- raw step output -->
    <pre class="raw-step" v-if="lastStep">{{ lastStepJson }}</pre>
  </main>
</template>


<script setup lang="ts">
import { ref, reactive, computed, onMounted, onUnmounted, nextTick } from 'vue'
import HeartbeatGraph from './HeartbeatGraph.vue'

// ── types ────────────────────────────────────────────────────────────────
interface StepEvent {
  step_index: number
  token_ids: number[]
  decoded_tokens: string[]
  mask_positions: number[]
  newly_revealed: number[]
}

interface HeartbeatPoint { x: number; y: number }

// ── state ────────────────────────────────────────────────────────────────
const online = ref(false)
const busy = ref(false)
const statusText = ref('Connecting…')
const genStatus = ref('')
const lastMsg = ref('—')
const userInput = ref('')

const arModel = ref<string | null>(null)
const diffusionModel = ref<string | null>(null)

const tokens = reactive<{ text: string; cls: string }[]>([])
const heartbeat = reactive<HeartbeatPoint[]>([])
let t0 = 0

const prevIds = reactive<{ ar: number[]; diffusion: number[] }>({ ar: [], diffusion: [] })

const lastStep = ref<StepEvent | null>(null)
const lastStepJson = computed(() => lastStep.value ? JSON.stringify(lastStep.value, null, 2) : '')

// params
const genLength = ref(64)
const steps = ref(64)
const temperature = ref(0)
const blockLength = ref(32)
const ttsEnabled = ref(true)

const constraintNote = computed(() => {
  const gl = genLength.value
  const bl = blockLength.value
  const nb = gl / bl
  const spb = steps.value / nb
  return `blocks=${nb}, steps/block=${spb}`
})

// ── heartbeat SVG path (reactive) ────────────────────────────────────────

// ── snap constraints ─────────────────────────────────────────────────────
function snapConstraints() {
  let gl = Math.max(8, genLength.value)
  let bl = Math.max(8, blockLength.value)
  let st = Math.max(1, steps.value)
  if (gl % bl !== 0) {
    // largest divisor of gl <= bl
    for (let d = bl; d >= 1; d--) {
      if (gl % d === 0) { bl = d; break }
    }
    blockLength.value = bl
  }
  const nb = gl / bl
  if (st % nb !== 0) {
    st = Math.max(nb, Math.round(st / nb) * nb)
    steps.value = st
  }
}

// ── backend polling ──────────────────────────────────────────────────────
let pollTimer: number | undefined

async function checkServer() {
  try {
    const r = await fetch('/api/status')
    const data = await r.json()
    const models: string[] = Object.keys(data.models ?? {})
    online.value = true
    if (!arModel.value) arModel.value = models.find(m => !m.toLowerCase().includes('llada')) ?? null
    if (!diffusionModel.value) diffusionModel.value = models.find(m => m.toLowerCase().includes('llada')) ?? null
    statusText.value = models.length ? models.join(', ') : 'no models loaded'
  } catch {
    online.value = false
    statusText.value = 'offline'
  }
}

onMounted(() => {
  checkServer()
  pollTimer = window.setInterval(checkServer, 5000)
  snapConstraints()
  // Warm up AudioContext on first user gesture so it's ready for TTS playback
  const warmUp = () => { getAudioCtx(); document.removeEventListener('click', warmUp) }
  document.addEventListener('click', warmUp)
})
onUnmounted(() => clearInterval(pollTimer))

// ── SSE streaming ────────────────────────────────────────────────────────
let abortCtrl: AbortController | null = null

async function streamGenerate(
  model: string,
  msg: string,
  prev: number[],
  onStep: (ev: StepEvent) => void,
): Promise<{ ok: boolean; finalIds: number[]; finalText: string }> {
  abortCtrl = new AbortController()
  let finalIds: number[] = []
  let finalText = ''
  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        messages: [{ role: 'user', content: msg }],
        prev_token_ids: prev,
        gen_length: genLength.value,
        steps: steps.value,
        block_length: blockLength.value,
        temperature: temperature.value,
        tts: false,   // TTS handled client-side via display queue
      }),
      signal: abortCtrl.signal,
    })
    if (!res.ok || !res.body) {
      genStatus.value = `⚠ Server error ${res.status}`
      return { ok: false, finalIds: [], finalText: '' }
    }
    const reader = res.body.getReader()
    const dec = new TextDecoder()
    let buf = ''
    let eventType = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += dec.decode(value, { stream: true })
      const lines = buf.split('\n')
      buf = lines.pop()!
      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed) continue
        if (trimmed.startsWith('event:')) {
          eventType = trimmed.slice(6).trim()
        } else if (trimmed.startsWith('data:')) {
          const data = JSON.parse(trimmed.slice(5).trim())
          if (eventType === 'step') onStep(data as StepEvent)
          else if (eventType === 'done') {
            finalIds = data.final_token_ids ?? []
            finalText = data.final_text ?? ''
            return { ok: true, finalIds, finalText }
          }
          else if (eventType === 'cancelled') { genStatus.value = 'Stopped.'; return { ok: false, finalIds: [], finalText: '' } }
          else if (eventType === 'error') { genStatus.value = `⚠ ${data.message}`; return { ok: false, finalIds: [], finalText: '' } }
        }
      }
    }
    return { ok: true, finalIds, finalText }
  } catch (e: any) {
    if (e.name === 'AbortError') { genStatus.value = 'Stopped.'; return { ok: false, finalIds: [], finalText: '' } }
    genStatus.value = `⚠ ${e.message}`
    return { ok: false, finalIds: [], finalText: '' }
  } finally {
    abortCtrl = null
  }
}

// ── TTS + display queue (Option C: step buffering) ───────────────────────
// Generation runs freely; TTS fetches fire in parallel; display waits per-step
// until its audio is ready, then plays + renders simultaneously.

const SKIP_TOKENS = new Set(['<|endoftext|>', '<|eot_id|>', '<eos>', '<s>', '</s>', '<pad>'])

let _audioCtx: AudioContext | null = null
function getAudioCtx(): AudioContext {
  if (!_audioCtx || _audioCtx.state === 'closed') _audioCtx = new AudioContext()
  if (_audioCtx.state === 'suspended') _audioCtx.resume()
  return _audioCtx
}

// Fetch TTS for a list of token texts in parallel; returns decoded AudioBuffers
async function fetchTtsBuffers(tokenTexts: string[]): Promise<AudioBuffer[]> {
  const valid = tokenTexts.filter(t => t.trim() && !SKIP_TOKENS.has(t.trim()))
  if (!valid.length) return []
  const ctx = getAudioCtx()
  const results = await Promise.all(valid.map(async text => {
    try {
      const res = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim() }),
      })
      if (!res.ok || res.status === 204) return null
      const ab = await res.arrayBuffer()
      return await ctx.decodeAudioData(ab)
    } catch { return null }
  }))
  return results.filter((b): b is AudioBuffer => b !== null)
}

// Schedule a set of already-decoded AudioBuffers to play simultaneously
function playAudioBuffers(bufs: AudioBuffer[], startOffset = 0.03): void {
  if (!bufs.length) return
  const ctx = getAudioCtx()
  const startTime = ctx.currentTime + startOffset
  for (const buf of bufs) {
    try {
      const src = ctx.createBufferSource()
      src.buffer = buf
      src.connect(ctx.destination)
      src.start(startTime)
    } catch { /* best-effort */ }
  }
}

// Display queue — each entry holds the step event, its TTS promise (already in-flight),
// and the render function to call once audio is ready.
interface QueuedStep {
  ev: StepEvent
  audioProm: Promise<AudioBuffer[]>
  renderFn: (ev: StepEvent) => void
}

let _displayQueue: QueuedStep[] = []
let _draining = false

function resetDisplayQueue(): void {
  _displayQueue = []
  _draining = false
}

// Serial drain: processes steps in arrival order, waiting for each step's TTS.
// New steps pushed during a drain are picked up by the running while-loop.
async function drainDisplayQueue(): Promise<void> {
  if (_draining) return
  _draining = true
  while (_displayQueue.length > 0) {
    const { ev, audioProm, renderFn } = _displayQueue.shift()!
    const bufs = await audioProm   // wait for this step's TTS (may already be done)
    playAudioBuffers(bufs)         // schedule audio — returns immediately
    pushHeartbeat(ev)
    renderFn(ev)                   // render tokens — same JS tick as audio scheduling
    scrollOutput()
  }
  _draining = false
}

// Enqueue a step: fire TTS immediately, push to queue, kick drain.
function enqueueStep(ev: StepEvent, renderFn: (ev: StepEvent) => void): void {
  const tokenTexts = ev.newly_revealed.map(i => ev.decoded_tokens[i])
  const audioProm = ttsEnabled.value
    ? fetchTtsBuffers(tokenTexts)
    : Promise.resolve([])
  _displayQueue.push({ ev, audioProm, renderFn })
  drainDisplayQueue()  // no-op if already draining; drain loop picks up new item
}

// ── token rendering helpers ──────────────────────────────────────────────
function renderAR(ev: StepEvent) {
  const revSet = new Set(ev.newly_revealed)
  tokens.length = 0
  for (let i = 0; i < ev.decoded_tokens.length; i++) {
    tokens.push({ text: ev.decoded_tokens[i] || '\u00a0', cls: revSet.has(i) ? 'tok-new' : 'tok-old' })
  }
}

function renderDiffusion(ev: StepEvent) {
  const maskSet = new Set(ev.mask_positions)
  const revSet = new Set(ev.newly_revealed)
  tokens.length = 0
  for (let i = 0; i < ev.decoded_tokens.length; i++) {
    if (maskSet.has(i)) tokens.push({ text: '▒', cls: 'tok-mask' })
    else tokens.push({ text: ev.decoded_tokens[i] || '\u00a0', cls: revSet.has(i) ? 'tok-new' : 'tok-old' })
  }
}

function pushHeartbeat(ev: StepEvent) {
  const elapsed = (performance.now() - t0) / 1000
  const unmasked = ev.decoded_tokens.length - ev.mask_positions.length
  heartbeat.push({ x: elapsed, y: unmasked })
  lastStep.value = ev
}

// ── actions ──────────────────────────────────────────────────────────────
const outputEl = ref<HTMLElement | null>(null)

function scrollOutput() {
  nextTick(() => { if (outputEl.value) outputEl.value.scrollTop = outputEl.value.scrollHeight })
}

async function runAR() {
  if (busy.value || !online.value || !arModel.value) return
  const msg = userInput.value.trim(); if (!msg) return
  userInput.value = ''
  busy.value = true
  lastMsg.value = msg
  genStatus.value = `⟳ ${arModel.value}…`
  tokens.length = 0
  heartbeat.length = 0
  lastStep.value = null
  t0 = performance.now()
  resetDisplayQueue()

  const { ok, finalIds } = await streamGenerate(arModel.value, msg, prevIds.ar, (ev) => {
    enqueueStep(ev, renderAR)
  })

  if (ok) { prevIds.ar = finalIds }
  genStatus.value = ok ? '✓ Done' : genStatus.value
  busy.value = false
}

async function runDiffuse() {
  if (busy.value || !online.value || !diffusionModel.value) return
  const msg = userInput.value.trim(); if (!msg) return
  userInput.value = ''
  busy.value = true
  lastMsg.value = msg
  genStatus.value = `⟳ ${diffusionModel.value}…`
  tokens.length = 0;
  heartbeat.length = 0
  lastStep.value = null
  t0 = performance.now()
  resetDisplayQueue()

  const { ok, finalIds } = await streamGenerate(diffusionModel.value, msg, prevIds.diffusion, (ev) => {
    enqueueStep(ev, renderDiffusion)
  })
  if (ok) { prevIds.diffusion = finalIds }
  genStatus.value = ok ? '✓ Done' : genStatus.value
  busy.value = false
}

async function runBoth() {
  if (busy.value || !online.value) return
  const msg = userInput.value.trim(); if (!msg) return
  userInput.value = ''
  busy.value = true
  lastMsg.value = msg
  tokens.length = 0
  heartbeat.length = 0
  lastStep.value = null
  t0 = performance.now()
  resetDisplayQueue()

  if (arModel.value) {
    genStatus.value = `⟳ ${arModel.value}…`
    const { ok, finalIds } = await streamGenerate(arModel.value, msg, prevIds.ar, (ev) => {
      enqueueStep(ev, renderAR)
    })
    if (ok) { prevIds.ar = finalIds }
    if (!ok) { busy.value = false; return }
  }

  // reset for diffusion pass
  tokens.length = 0; heartbeat.length = 0; lastStep.value = null; t0 = performance.now()
  resetDisplayQueue()

  if (diffusionModel.value) {
    genStatus.value = `⟳ ${diffusionModel.value}…`
    const { ok, finalIds } = await streamGenerate(diffusionModel.value, msg, prevIds.diffusion, (ev) => {
      enqueueStep(ev, renderDiffusion)
    })
    if (ok) { prevIds.diffusion = finalIds }
    genStatus.value = ok ? '✓ Done' : genStatus.value
  }
  busy.value = false
}

async function stopOrClear() {
  if (busy.value) {
    abortCtrl?.abort()
    resetDisplayQueue()
    try { await fetch('/api/stop', { method: 'POST' }) } catch {}
  } else {
    tokens.length = 0
    heartbeat.length = 0
    lastStep.value = null
    lastMsg.value = '—'
    genStatus.value = ''
    prevIds.ar = []; prevIds.diffusion = []
  }
}

function onEnter() { runBoth() }
</script>

<style>



* {
  margin: 0; padding: 0; box-sizing: border-box;
}
body {
  background: #0f172a; color: #e2e8f0; font-family: system-ui, sans-serif;
}
header {
  display: flex; align-items: center; gap: 8px; padding: 8px 16px; background: #1e293b;
}
.title {
  font-size: 18px; font-weight: 700;
}
.dot {
  font-size: 14px;
}
.dot.on {
  color: #4ade80;
}
.dot.off {
  color: #f87171;
}
.status-text {
  font-size: 12px; color: #94a3b8;
}

main {
  max-width: 900px;
  margin: 0 auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.last-msg {
  font-size: 13px;
  color: #cbd5e1;
  background: #1e293b;
  border-radius: 6px;
  padding: 6px 12px;
}


.output {
  min-height: 200px;
  max-height: 400px;
  overflow-y: auto;
  background: #020617;
  border: 1px solid #334155;
  border-radius: 6px;
  padding: 10px;
  font-family: monospace;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-all;
}
.tok {
  padding: 1px 2px;
  border-radius: 3px;
  background: rgba(255,255,255,0.05);
  display: inline;
}
.tok-new {
  color: #fbbf24;
}
.tok-old {
  color: #94a3b8;
}
.tok-mask {
  color: #475569;
}

.params {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
  font-size: 13px;
}
.params label {
  display: flex;
  align-items: center;
  gap: 4px;
}
.params input {
  width: 64px;
  background: #1e293b;
  border: 1px solid #334155;
  color: #e2e8f0;
  border-radius: 4px;
  padding: 3px 6px;
}
.note {
  font-size: 11px; color: #eab308;
}

.controls {
  display: flex; gap: 6px;
}
.msg-input {
  flex: 1;
  background: #1e293b;
  border: 1px solid #334155;
  color: #e2e8f0;
  border-radius: 4px;
  padding: 6px 10px;
  font-size: 14px;
}
.btn {
  padding: 6px 14px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 600;
  color: #fff;
}
.btn:disabled {
  opacity: 0.4;
  cursor: default;
}
.btn.ar {
  background: #7c3aed;
}
.btn.diff {
  background: #0d9488;
}
.btn.stop {
  background: #dc2626;
}
.btn.tts-on {
  background: #1d4ed8;
}
.btn.tts-off {
  background: #334155;
}

.gen-status {
  font-size: 12px; color: #64748b;
}

.raw-step {
  background: #020617;
  border: 1px solid #334155;
  border-radius: 6px;
  padding: 10px;
  font-family: monospace;
  font-size: 11px;
  color: #64748b;
  max-height: 200px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-all;
}
</style>
