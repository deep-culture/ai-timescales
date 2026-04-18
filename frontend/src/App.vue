<template>
  <!-- ── Login overlay ─────────────────────────────────────────────────── -->
  <div v-if="loginRequired && !loggedIn" class="login-overlay">
    <form class="login-box" @submit.prevent="doLogin">
      <h2>AI Timescales</h2>
      <p v-if="loginError" class="login-error">{{ loginError }}</p>
      <label>Username<input v-model="loginUser" type="text" autocomplete="username" /></label>
      <label>Password<input v-model="loginPass" type="password" autocomplete="current-password" /></label>
      <button type="submit" :disabled="loginBusy">Sign in</button>
    </form>
  </div>

  <template v-else>
  <header>
    <span class="title"><strong>AI timescales</strong></span>
    <span :class="['dot', online ? 'on' : 'off']">●</span>
    <span class="status-text">{{ statusText }}</span>
  </header>

  <main>

    <!-- params -->
    <div class="control-box intro-text">
      <div>How can we listen to the rhythms in deep learning?</div>
      <div>How do different operations traverse timescales?</div>
      <div>What rhythms ‘get through’ to other temporal strata?</div>
    </div>
    <div class="control-box params">
      <span><label>Gen length <input type="number" v-model.number="genLength" min="8" max="256" step="8" @change="snapConstraints()" /></label></span>
      <span><label>Steps <input type="number" v-model.number="steps" min="1" max="256" step="1" @change="snapConstraints()" /></label></span>
      <span><label>Temp <input type="number" v-model.number="temperature" min="0" max="2" step="0.1" /></label></span>
      <span><label>Block len <input type="number" v-model.number="blockLength" min="8" max="256" step="8" @change="snapConstraints()" /></label></span>
      <span class="note">{{ constraintNote }}</span>
    </div>
    <!-- scales   -->
    <div class="control-box scales">
      <span class="scales-header">Timescale</span>
      <button :class="['btn', 'toggle', timescale === 'inference' ? 'toggle-active' : '']" @click="switchTimescale('inference')">Inference</button>
      <button :class="['btn', 'toggle', timescale === 'attention' ? 'toggle-active' : '']" @click="switchTimescale('attention')">Attention</button>
    </div>
    <!-- input + buttons -->
    <div class="control-box controls last-controls">
      <div>
        <input class="prompt-input" v-model="userInput" placeholder="Insert prompt" @keydown.enter="onEnter" />
      </div>
      <div class="btn-row">
        <template v-if="timescale === 'inference'">
          <button class="btn ar" :disabled="busy || !online || !arModel" @click="runAR">Autoregress</button>
          <button class="btn diff" :disabled="busy || !online || !diffusionModel" @click="runDiffuse">Diffuse</button>
        </template>
        <template v-else>
          <button class="btn ar" :disabled="busy || !online || !arModel" @click="nextARStep">Next autoregression</button>
          <button class="btn diff" :disabled="busy || !online || !diffusionModel" @click="nextDiffStep">Next denoise</button>
        </template>
        <button class="btn stop" :disabled="!busy" @click="stopOrClear" title="Stop inference">&#9632;</button>
      </div>
      <!-- head selector — only in attention timescale -->
      <div v-if="timescale === 'attention'" class="head-row">
        <div><h2>Select final attention heads</h2></div>
        <div>
          <button
            :class="['head-btn', headAvg ? 'head-active' : '']"
            @click="toggleAvg()"
          >avg</button>
          <button
            v-for="h in nHeads" :key="h - 1"
            :class="['head-btn', (!headAvg && selectedHeads.has(h - 1)) ? 'head-active' : '']"
            @click="toggleHead(h - 1)"
            :title="headVoice(h - 1)"
          >{{ h - 1 }}</button>
          </div>
      </div>
    </div>

    <div class="gen-status">{{ genStatus }}</div>

    <!-- heartbeat graph -->
    <HeartbeatGraph
      v-if="timescale !== 'attention'"
      :points="heartbeat"
      :netPoints="netPoints"
      :stepTimes="stepTimes"
      :modelLabel="modelLabel"
    />

    <!-- single merged output box -->
    <div class="output" ref="outputEl">
      <span v-for="(tok, i) in tokens" :key="i"
        :class="['tok', tok.cls, isSelectedToken(i) ? 'tok-selected' : '']"
        @click="onTokenClick(i)"
        :style="{
          cursor: (timescale === 'attention' && isDiffAttnMode) ? 'pointer' : 'default',
          filter: (timescale === 'attention' && tokenBlurs[i] !== undefined)
            ? `blur(${tokenBlurs[i].toFixed(2)}px)` : 'none',
        }"
      >{{ tok.text }}</span>
    </div>

    <!-- raw step output -->
    <pre class="raw-step" v-if="lastStep">{{ lastStepJson }}</pre>
    <footer>
      <a href="https://deep-culture.org">
        <img src="./assets/deep-culture.png">
      </a>
    </footer>
  </main>
  </template>
</template>


<script setup lang="ts">
import { ref, reactive, computed, onMounted, onUnmounted, nextTick } from 'vue'
import HeartbeatGraph from './components/HeartbeatGraph.vue'

// ── types ────────────────────────────────────────────────────────────────
interface StepEvent {
  step_index: number
  token_ids: number[]
  decoded_tokens: string[]
  mask_positions: number[]
  newly_revealed: number[]
  attention?: number[] | number[][]          // mean: 1D (AR) or 2D (diffusion)
  attention_heads?: number[][] | number[][][] // per-head: (n_heads,T) or (n_heads,gen,gen)
  n_heads?: number
  prompt_length?: number
}

interface HeartbeatPoint { x: number; y: number }

// ── auth ─────────────────────────────────────────────────────────────────
const loginRequired = ref(false)
const loggedIn = ref(false)
const authToken = ref<string | null>(sessionStorage.getItem('auth_token'))
const loginUser = ref('')
const loginPass = ref('')
const loginError = ref('')
const loginBusy = ref(false)

if (authToken.value) loggedIn.value = true

function authHeaders(): Record<string, string> {
  if (authToken.value) return { 'Authorization': `Bearer ${authToken.value}` }
  return {}
}

async function doLogin() {
  loginBusy.value = true
  loginError.value = ''
  try {
    const res = await fetch('/api/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: loginUser.value, password: loginPass.value }),
    })
    if (!res.ok) { loginError.value = 'Invalid username or password.'; return }
    const data = await res.json()
    authToken.value = data.token
    sessionStorage.setItem('auth_token', data.token)
    loggedIn.value = true
    loginPass.value = ''
    checkServer()
  } catch {
    loginError.value = 'Connection error.'
  } finally {
    loginBusy.value = false
  }
}

// ── state ────────────────────────────────────────────────────────────────
const online = ref(false)
const busy = ref(false)
const statusText = ref('Connecting…')
const genStatus = ref('')
const userInput = ref('')

const arModel = ref<string | null>(null)
const diffusionModel = ref<string | null>(null)

const tokens = reactive<{ text: string; cls: string }[]>([])
const heartbeat = reactive<HeartbeatPoint[]>([])
const netPoints = reactive<HeartbeatPoint[]>([])
const stepTimes = reactive<number[]>([])
let t0 = 0
let _netCount = 0

// ── network request tracker ──────────────────────────────────────────────
function recordNetRequest() {
  if (t0 === 0) return
  const elapsed = (performance.now() - t0) / 1000
  _netCount++
  netPoints.push({ x: elapsed, y: _netCount })
}

// ── model badge ───────────────────────────────────────────────────────────
const _tokensPerSecHistory = reactive<number[]>([])
let _lastStepTime = 0
const _currentRunModel = ref<string | null>(null)
const _currentRunType = ref<'AR' | 'diffusion' | null>(null)

const modelLabel = computed(() => {
  const m = _currentRunModel.value
  if (!m) return ''
  const short = m.split('/').pop() ?? m
  const type = _currentRunType.value ?? ''
  const avg = _tokensPerSecHistory.length
    ? (_tokensPerSecHistory.reduce((a, b) => a + b, 0) / _tokensPerSecHistory.length).toFixed(1)
    : null
  return avg ? `${short} · ${type} · ~${avg} tok/s` : `${short} · ${type}`
})

function resetRunMetrics(model: string, type: 'AR' | 'diffusion') {
  _currentRunModel.value = model
  _currentRunType.value = type
  _tokensPerSecHistory.length = 0
  _lastStepTime = 0
}

function recordStepTempo(newTokenCount: number) {
  const now = performance.now() / 1000
  if (_lastStepTime > 0 && newTokenCount > 0) {
    const dt = now - _lastStepTime
    if (dt > 0) _tokensPerSecHistory.push(newTokenCount / dt)
  }
  _lastStepTime = now
  stepTimes.push((performance.now() - t0) / 1000)
}

const prevIds = reactive<{ ar: number[]; diffusion: number[] }>({ ar: [], diffusion: [] })

const lastStep = ref<StepEvent | null>(null)
const lastStepJson = computed(() => lastStep.value ? JSON.stringify(lastStep.value, null, 2) : '')

// params
const genLength = ref(64)
const steps = ref(64)
const temperature = ref(0)
const blockLength = ref(32)
const ttsEnabled = ref(true)
const ttsVolume = ref(1)

// ── timescale mode ───────────────────────────────────────────────────────
const timescale = ref<'inference' | 'attention'>('inference')
const selectedTokenIdx = ref(0)

// ── head selection (multi-select) ────────────────────────────────────────
const headAvg = ref(true)
const selectedHeads = reactive(new Set<number>())
const nHeads = ref(32)
const availableVoices = ref<string[]>(['af_heart'])
const tokenBlurs = reactive<number[]>([])

// Fixed voice palette: index 0 = avg voice, indices 1-4 = head voices (cycling)
const DEFAULT_VOICES = ['af_heart', 'af_nicole', 'am_michael', 'bf_alice', 'bm_fable']

// ── derived state ─────────────────────────────────────────────────────────
const isDiffAttnMode = computed(() =>
  timescale.value === 'attention' && (lastStep.value?.mask_positions?.length ?? 0) > 0
)

function isSelectedToken(i: number): boolean {
  if (timescale.value !== 'attention') return false
  if (isDiffAttnMode.value) return i === selectedTokenIdx.value
  return i === tokens.length - 1
}

async function fetchVoices() {
  try {
    const r = await fetch('/api/voices', { headers: authHeaders() })
    if (r.ok) availableVoices.value = await r.json()
  } catch { /* use default */ }
}

// Head h → voice index 1..4 (cycling), title shown as tooltip
function headVoice(h: number): string {
  return DEFAULT_VOICES[1 + (h % 4)]
}

// Avg always uses the first voice
function avgVoice(): string { return DEFAULT_VOICES[0] }

function toggleAvg() {
  headAvg.value = true
  selectedHeads.clear()
  _recomputeAfterHeadChange()
}

function toggleHead(h: number) {
  if (headAvg.value) {
    headAvg.value = false
    selectedHeads.clear()
    selectedHeads.add(h)
  } else if (selectedHeads.has(h)) {
    selectedHeads.delete(h)
    if (selectedHeads.size === 0) headAvg.value = true
  } else {
    selectedHeads.add(h)
  }
  _recomputeAfterHeadChange()
}

function _recomputeAfterHeadChange() {
  if (!lastStep.value || timescale.value !== 'attention') return
  if (isDiffAttnMode.value) {
    updateDiffusionBlurs(lastStep.value)
    playDiffusionAttentionTTS(lastStep.value)
  } else {
    updateARBlurs(lastStep.value, arAttnTokens.length)
  }
}

// ── effective attention row (averaged over selection, used for blur) ──────
function getEffectiveAttnRow(ev: StepEvent, rowIdx?: number): number[] {
  const isDiff = (ev.mask_positions?.length ?? 0) > 0
  const useMean = headAvg.value || selectedHeads.size === 0

  if (useMean || !ev.attention_heads) {
    if (!isDiff) return (ev.attention as number[]) ?? []
    return (ev.attention as number[][])?.[rowIdx ?? 0] ?? []
  }

  const heads = [...selectedHeads]
  if (!isDiff) {
    const rows = heads.map(h => (ev.attention_heads as number[][])[h] ?? [])
    if (!rows.length) return (ev.attention as number[]) ?? []
    const len = rows[0].length
    return Array.from({ length: len }, (_, i) =>
      rows.reduce((s, r) => s + (r[i] ?? 0), 0) / rows.length)
  } else {
    const rows = heads.map(h => (ev.attention_heads as number[][][])[h]?.[rowIdx ?? 0] ?? [])
    if (!rows.length) return (ev.attention as number[][])?.[rowIdx ?? 0] ?? []
    const len = rows[0].length
    return Array.from({ length: len }, (_, i) =>
      rows.reduce((s, r) => s + (r[i] ?? 0), 0) / rows.length)
  }
}

// ── single head attention row (not averaged, used for per-voice TTS) ─────
function getSingleHeadRow(ev: StepEvent, h: number, rowIdx?: number): number[] {
  const isDiff = rowIdx !== undefined
  if (!ev.attention_heads) return getEffectiveAttnRow(ev, rowIdx)
  if (!isDiff) return (ev.attention_heads as number[][])[h] ?? (ev.attention as number[]) ?? []
  return (ev.attention_heads as number[][][])[h]?.[rowIdx] ??
         (ev.attention as number[][])?.[rowIdx] ?? []
}

// ── attention layers for TTS: one entry per active head (or avg) ──────────
// rowIdx: undefined = AR (full last row), number = diffusion (gen-portion row idx)
function buildAttnLayers(ev: StepEvent, rowIdx?: number): Array<{ voice: string; row: number[] }> {
  if (headAvg.value || selectedHeads.size === 0) {
    return [{ voice: avgVoice(), row: getEffectiveAttnRow(ev, rowIdx) }]
  }
  return [...selectedHeads].map(h => ({ voice: headVoice(h), row: getSingleHeadRow(ev, h, rowIdx) }))
}

const MAX_BLUR = 8

function updateARBlurs(ev: StepEvent, genCount: number) {
  const attnRow = getEffectiveAttnRow(ev)
  const pl = Math.max(0, attnRow.length - genCount)
  const scores = Array.from({ length: genCount }, (_, i) => attnRow[pl + i] ?? 0)
  const maxS = Math.max(...scores, 1e-9)
  tokenBlurs.length = 0
  for (let i = 0; i < genCount; i++) {
    tokenBlurs.push(i === genCount - 1 ? 0 : MAX_BLUR * (1 - scores[i] / maxS))
  }
}

function updateDiffusionBlurs(ev: StepEvent) {
  const maskSet = new Set(ev.mask_positions)
  const total = ev.decoded_tokens.length
  const selIdx = selectedTokenIdx.value
  const attnRow = getEffectiveAttnRow(ev, selIdx)
  const nonMaskMax = Math.max(...attnRow.filter((_, i) => !maskSet.has(i)), 1e-9)
  tokenBlurs.length = 0
  for (let i = 0; i < total; i++) {
    if (i === selIdx) { tokenBlurs.push(0); continue }
    if (maskSet.has(i)) { tokenBlurs.push(MAX_BLUR); continue }
    tokenBlurs.push(MAX_BLUR * (1 - (attnRow[i] ?? 0) / nonMaskMax))
  }
}

// AR attention mode state
const arAttnTokens = reactive<string[]>([])   // accumulated generated token texts
const arAttnIds = ref<number[]>([])           // full token IDs for continue_only
const arLastAttnRow = ref<number[] | null>(null)

// Diffusion attention mode state
const diffStepBuffer = reactive<StepEvent[]>([])
const diffStepIdx = ref(0)
const diffGenerationDone = ref(false)
const diffCurrentAttn = ref<number[][] | null>(null)

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
    const r = await fetch('/api/status', { headers: authHeaders() })
    if (r.status === 401) {
      online.value = false
      statusText.value = 'login required'
      loggedIn.value = false
      return
    }
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

onMounted(async () => {
  // Check if login is required
  try {
    const r = await fetch('/api/auth-config')
    const data = await r.json()
    loginRequired.value = data.required
    if (!data.required) loggedIn.value = true
  } catch { /* if backend is down, just proceed */ }

  checkServer()
  fetchVoices()
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
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify({
        model,
        messages: [{ role: 'user', content: msg }],
        prev_token_ids: prev,
        gen_length: genLength.value,
        steps: steps.value,
        block_length: blockLength.value,
        temperature: temperature.value,
        tts: false,   // TTS handled client-side via display queue
        return_attention: timescale.value === 'attention',
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

// Fetch TTS for a list of token texts; returns decoded AudioBuffers with their pan positions
async function fetchTtsBuffers(tokenTexts: string[], pans: number[], voice = 'af_heart'): Promise<{ buffer: AudioBuffer; pan: number }[]> {
  const ctx = getAudioCtx()
  const results = await Promise.all(tokenTexts.map(async (text, i) => {
    if (!text.trim() || SKIP_TOKENS.has(text.trim())) return null
    try {
      recordNetRequest()
      const res = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: JSON.stringify({ text: text.trim(), voice }),
      })
      if (!res.ok || res.status === 204) return null
      const ab = await res.arrayBuffer()
      const buffer = await ctx.decodeAudioData(ab)
      return { buffer, pan: pans[i] ?? 0 }
    } catch { return null }
  }))
  return results.filter((b): b is { buffer: AudioBuffer; pan: number } => b !== null)
}

// Schedule a set of already-decoded AudioBuffers to play simultaneously with individual panning
function playAudioBuffers(items: { buffer: AudioBuffer; pan: number }[], startOffset = 0.03): void {
  if (!items.length) return
  const ctx = getAudioCtx()
  const startTime = ctx.currentTime + startOffset
  const gainNode = ctx.createGain()
  gainNode.gain.value = ttsVolume.value
  gainNode.connect(ctx.destination)
  for (const { buffer, pan } of items) {
    try {
      const src = ctx.createBufferSource()
      src.buffer = buffer
      const panner = ctx.createStereoPanner()
      panner.pan.value = pan
      src.connect(panner)
      panner.connect(gainNode)
      src.start(startTime)
    } catch { /* best-effort */ }
  }
}

// Display queue — audioProm is started lazily (1-step lookahead only)
interface QueuedStep {
  ev: StepEvent
  renderFn: (ev: StepEvent) => void
  audioProm?: Promise<{ buffer: AudioBuffer; pan: number }[]>
}

let _displayQueue: QueuedStep[] = []
let _draining = false

function resetDisplayQueue(): void {
  _displayQueue = []
  _draining = false
}

function _buildAudioProm(ev: StepEvent): Promise<{ buffer: AudioBuffer; pan: number }[]> {
  if (!ttsEnabled.value) return Promise.resolve([])
  const totalLen = genLength.value
  const tokenTexts = ev.newly_revealed.map(i => ev.decoded_tokens[i])
  const pans = ev.newly_revealed.map(i => {
    const t = totalLen > 1 ? i / (totalLen - 1) : 0.5
    return (t * 2) - 1
  })
  return fetchTtsBuffers(tokenTexts, pans)
}

// Serial drain: 1-step TTS lookahead.
// Starts TTS for step N while rendering step N-1 — at most one request in-flight,
// so burst generation never queues up parallel TTS calls.
async function drainDisplayQueue(): Promise<void> {
  if (_draining) return
  _draining = true
  // Prime the first item
  if (_displayQueue.length > 0 && !_displayQueue[0].audioProm) {
    _displayQueue[0].audioProm = _buildAudioProm(_displayQueue[0].ev)
  }
  while (_displayQueue.length > 0) {
    const item = _displayQueue.shift()!
    if (!item.audioProm) item.audioProm = _buildAudioProm(item.ev)
    // Start TTS for the next item now (pipeline: overlaps with current await)
    if (_displayQueue.length > 0 && !_displayQueue[0].audioProm) {
      _displayQueue[0].audioProm = _buildAudioProm(_displayQueue[0].ev)
    }
    const items = await item.audioProm
    pushHeartbeat(item.ev)
    item.renderFn(item.ev)
    scrollOutput()
    playAudioBuffers(items)
    await nextTick()
    await new Promise(r => setTimeout(r, 0))
  }
  _draining = false
}

// Enqueue a step: do NOT fire TTS here — the drain controls timing.
function enqueueStep(ev: StepEvent, renderFn: (ev: StepEvent) => void): void {
  _displayQueue.push({ ev, renderFn })
  drainDisplayQueue()
}


// ── timescale switching ──────────────────────────────────────────────────
function switchTimescale(mode: 'inference' | 'attention') {
  timescale.value = mode
  // Reset state when switching
  tokens.length = 0
  heartbeat.length = 0; netPoints.length = 0; _netCount = 0; stepTimes.length = 0
  lastStep.value = null
  genStatus.value = ''
  resetDisplayQueue()
  // Reset attention-mode state
  arAttnTokens.length = 0
  arAttnIds.value = []
  arLastAttnRow.value = null
  diffStepBuffer.length = 0
  diffStepIdx.value = 0
  diffGenerationDone.value = false
  diffCurrentAttn.value = null
  selectedTokenIdx.value = 0
  tokenBlurs.length = 0
  headAvg.value = true
  selectedHeads.clear()
}

// ── pan helper ───────────────────────────────────────────────────────────
function panForIndex(i: number, total: number): number {
  const t = total > 1 ? i / (total - 1) : 0.5
  return (t * 2) - 1
}

// ── volume-aware TTS playback ────────────────────────────────────────────
async function fetchTtsBuffersWithVolumes(
  tokenTexts: string[],
  pans: number[],
  volumes: number[],
  voices?: string[],
): Promise<{ buffer: AudioBuffer; pan: number; volume: number }[]> {
  const ctx = getAudioCtx()
  const defaultVoice = availableVoices.value[0] ?? 'af_heart'
  const results = await Promise.all(tokenTexts.map(async (text, i) => {
    if (!text.trim() || SKIP_TOKENS.has(text.trim())) return null
    try {
      recordNetRequest()
      const voice = voices?.[i] ?? defaultVoice
      const res = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: JSON.stringify({ text: text.trim(), voice }),
      })
      if (!res.ok || res.status === 204) return null
      const ab = await res.arrayBuffer()
      const buffer = await ctx.decodeAudioData(ab)
      return { buffer, pan: pans[i] ?? 0, volume: volumes[i] ?? 1 }
    } catch { return null }
  }))
  return results.filter((b): b is { buffer: AudioBuffer; pan: number; volume: number } => b !== null)
}

function playAudioBuffersWithVolumes(items: { buffer: AudioBuffer; pan: number; volume: number }[], startOffset = 0.03): number {
  if (!items.length) return 0
  const ctx = getAudioCtx()
  const startTime = ctx.currentTime + startOffset
  let maxDuration = 0
  for (const { buffer, pan, volume } of items) {
    try {
      const src = ctx.createBufferSource()
      src.buffer = buffer
      const gainNode = ctx.createGain()
      gainNode.gain.value = volume * ttsVolume.value
      const panner = ctx.createStereoPanner()
      panner.pan.value = pan
      src.connect(panner)
      panner.connect(gainNode)
      gainNode.connect(ctx.destination)
      src.start(startTime)
      maxDuration = Math.max(maxDuration, buffer.duration)
    } catch { /* best-effort */ }
  }
  return maxDuration
}

// ── single-step fetch helper ─────────────────────────────────────────────
async function fetchSteps(body: Record<string, unknown>): Promise<{ steps: StepEvent[]; finalIds: number[]; finalText: string }> {
  const res = await fetch('/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(body),
  })
  if (!res.ok || !res.body) return { steps: [], finalIds: [], finalText: '' }
  const reader = res.body.getReader()
  const dec = new TextDecoder()
  let buf = ''
  let eventType = ''
  const collectedSteps: StepEvent[] = []
  let finalIds: number[] = []
  let finalText = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    const lines = buf.split('\n')
    buf = lines.pop()!
    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed) continue
      if (trimmed.startsWith('event:')) eventType = trimmed.slice(6).trim()
      else if (trimmed.startsWith('data:')) {
        const data = JSON.parse(trimmed.slice(5).trim())
        if (eventType === 'step') collectedSteps.push(data as StepEvent)
        else if (eventType === 'done') { finalIds = data.final_token_ids ?? []; finalText = data.final_text ?? '' }
      }
    }
  }
  return { steps: collectedSteps, finalIds, finalText }
}

// ── AR attention mode ────────────────────────────────────────────────────
async function nextARStep() {
  if (busy.value || !online.value || !arModel.value) return
  const msg = userInput.value.trim()
  if (!msg && arAttnIds.value.length === 0) return
  busy.value = true
  genStatus.value = '⟳ next AR token…'

  const isFirst = arAttnIds.value.length === 0
  const { steps: stepEvts, finalIds } = await fetchSteps({
    model: arModel.value,
    messages: isFirst ? [{ role: 'user', content: msg }] : [],
    prev_token_ids: arAttnIds.value,
    gen_length: 1,
    steps: 1,
    block_length: 1,
    temperature: temperature.value,
    tts: false,
    return_attention: true,
    continue_only: !isFirst,
  })

  if (!stepEvts.length) { busy.value = false; genStatus.value = '⚠ no step'; return }
  const ev = stepEvts[0]
  arAttnIds.value = finalIds
  const newText = ev.decoded_tokens[0] ?? ''
  arAttnTokens.push(newText)
  arLastAttnRow.value = (ev.attention as number[]) ?? null
  lastStep.value = ev

  // Render all accumulated tokens (last = selected, always sharp)
  const total = arAttnTokens.length
  tokens.length = 0
  for (let i = 0; i < arAttnTokens.length; i++) {
    tokens.push({
      text: arAttnTokens[i] || '\u00a0',
      cls: i === arAttnTokens.length - 1 ? 'tok-new' : 'tok-old',
    })
  }
  // Update blur based on current head selection
  updateARBlurs(ev, total)
  scrollOutput()

  // TTS: play new token first with avg voice (always sharp / selected)
  const newBufs = await fetchTtsBuffersWithVolumes(
    [newText], [panForIndex(total - 1, total)], [1], [avgVoice()])
  let dur = 0
  if (newBufs.length) {
    dur = playAudioBuffersWithVolumes(newBufs)
    await new Promise(r => setTimeout(r, dur * 1000 + 80))
  }

  // Replay previous tokens — one TTS layer per active head (or avg), all fired simultaneously
  if (total > 1) {
    const prevCount = total - 1
    const prevTexts = arAttnTokens.slice(0, prevCount)
    const prevPans = prevTexts.map((_, i) => panForIndex(i, total))
    const layers = buildAttnLayers(ev)   // each layer: {voice, row}
    const allBufs = await Promise.all(layers.map(async ({ voice, row }) => {
      const pl = Math.max(0, row.length - total)
      const rawVols = Array.from({ length: prevCount }, (_, i) => row[pl + i] ?? 0)
      const maxV = Math.max(...rawVols, 1e-9)
      const normVols = rawVols.map(v => v / maxV)
      return fetchTtsBuffersWithVolumes(prevTexts, prevPans, normVols, prevTexts.map(() => voice))
    }))
    for (const bufs of allBufs) playAudioBuffersWithVolumes(bufs)
  }

  genStatus.value = `✓ token ${total}`
  busy.value = false
}

// ── Diffusion attention mode ─────────────────────────────────────────────
async function nextDiffStep() {
  if (busy.value || !online.value || !diffusionModel.value) return

  // If buffer is exhausted or empty, start a new generation
  if (diffStepIdx.value >= diffStepBuffer.length) {
    const msg = userInput.value.trim()
    if (!msg) return
    busy.value = true
    genStatus.value = '⟳ generating all steps…'
    diffStepBuffer.length = 0
    diffStepIdx.value = 0
    diffGenerationDone.value = false
    selectedTokenIdx.value = 0

    const { steps: allSteps } = await fetchSteps({
      model: diffusionModel.value,
      messages: [{ role: 'user', content: msg }],
      prev_token_ids: [],
      gen_length: genLength.value,
      steps: steps.value,
      block_length: blockLength.value,
      temperature: temperature.value,
      tts: false,
      return_attention: true,
    })

    for (const s of allSteps) diffStepBuffer.push(s)
    diffGenerationDone.value = true

    if (!diffStepBuffer.length) { busy.value = false; genStatus.value = '⚠ no steps'; return }
  } else {
    busy.value = true
  }

  // Show the current step
  const ev = diffStepBuffer[diffStepIdx.value]
  diffStepIdx.value++
  diffCurrentAttn.value = (ev.attention as number[][] | undefined) ?? null
  lastStep.value = ev

  // Render diffusion tokens + update blur
  renderDiffusion(ev)
  updateDiffusionBlurs(ev)
  scrollOutput()

  // TTS: play selected token first, then all others with attention-mapped volume
  await playDiffusionAttentionTTS(ev)

  genStatus.value = `step ${diffStepIdx.value}/${diffStepBuffer.length}`
  busy.value = false
}

async function playDiffusionAttentionTTS(ev: StepEvent) {
  const maskSet = new Set(ev.mask_positions)
  const sel = selectedTokenIdx.value
  const total = ev.decoded_tokens.length

  // Selected token: play at full volume with avg voice (always sharp)
  if (!maskSet.has(sel)) {
    const selText = ev.decoded_tokens[sel]
    const selBufs = await fetchTtsBuffersWithVolumes(
      [selText], [panForIndex(sel, total)], [1], [avgVoice()])
    if (selBufs.length) {
      const dur = playAudioBuffersWithVolumes(selBufs)
      await new Promise(r => setTimeout(r, dur * 1000 + 80))
    }
  }

  // Build the list of other visible tokens
  const otherIdxs: number[] = [], otherTexts: string[] = [], otherPans: number[] = []
  for (let i = 0; i < total; i++) {
    if (i === sel || maskSet.has(i)) continue
    otherIdxs.push(i)
    otherTexts.push(ev.decoded_tokens[i])
    otherPans.push(panForIndex(i, total))
  }
  if (!otherTexts.length) return

  // One TTS layer per active head (or single avg layer), all fired simultaneously
  const layers = buildAttnLayers(ev, sel)
  const allBufs = await Promise.all(layers.map(async ({ voice, row }) => {
    const rawVols = otherIdxs.map(i => row[i] ?? 0)
    const maxV = Math.max(...rawVols, 1e-9)
    const normVols = rawVols.map(v => v / maxV)
    return fetchTtsBuffersWithVolumes(otherTexts, otherPans, normVols, otherTexts.map(() => voice))
  }))
  for (const bufs of allBufs) playAudioBuffersWithVolumes(bufs)
}

// ── token click handler ──────────────────────────────────────────────────
function onTokenClick(i: number) {
  // Only interactive for diffusion attention mode
  if (!isDiffAttnMode.value) return
  selectedTokenIdx.value = i
  if (lastStep.value) {
    updateDiffusionBlurs(lastStep.value)
    playDiffusionAttentionTTS(lastStep.value)
  }
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
  recordStepTempo(ev.newly_revealed.length)
  if (ev.n_heads) nHeads.value = ev.n_heads
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
  busy.value = true
  genStatus.value = `⟳ ${arModel.value}…`
  tokens.length = 0
  heartbeat.length = 0; netPoints.length = 0; _netCount = 0; stepTimes.length = 0
  lastStep.value = null
  t0 = performance.now()
  resetDisplayQueue()
  resetRunMetrics(arModel.value, 'AR')

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
  busy.value = true
  genStatus.value = `⟳ ${diffusionModel.value}…`
  tokens.length = 0
  heartbeat.length = 0; netPoints.length = 0; _netCount = 0; stepTimes.length = 0
  lastStep.value = null
  t0 = performance.now()
  resetDisplayQueue()
  resetRunMetrics(diffusionModel.value, 'diffusion')

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
  busy.value = true
  tokens.length = 0
  heartbeat.length = 0; netPoints.length = 0; _netCount = 0; stepTimes.length = 0
  lastStep.value = null
  t0 = performance.now()
  resetDisplayQueue()

  if (arModel.value) {
    genStatus.value = `⟳ ${arModel.value}…`
    resetRunMetrics(arModel.value, 'AR')
    const { ok, finalIds } = await streamGenerate(arModel.value, msg, prevIds.ar, (ev) => {
      enqueueStep(ev, renderAR)
    })
    if (ok) { prevIds.ar = finalIds }
    if (!ok) { busy.value = false; return }
  }

  // reset for diffusion pass
  tokens.length = 0; heartbeat.length = 0; netPoints.length = 0; _netCount = 0; stepTimes.length = 0; lastStep.value = null; t0 = performance.now()
  resetDisplayQueue()

  if (diffusionModel.value) {
    genStatus.value = `⟳ ${diffusionModel.value}…`
    resetRunMetrics(diffusionModel.value, 'diffusion')
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
    try { await fetch('/api/stop', { method: 'POST', headers: authHeaders() }) } catch {}
  } else {
    tokens.length = 0
    heartbeat.length = 0; netPoints.length = 0; _netCount = 0; stepTimes.length = 0
    lastStep.value = null
    genStatus.value = ''
    prevIds.ar = []; prevIds.diffusion = []
  }
}

function onEnter() {
  if (timescale.value === 'inference') runBoth()
  else nextARStep()  // Enter in attention mode → next AR token
}
</script>

<style>
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@300;400;600;700&display=swap');

:root {
  --font-sans: 'Lexend', system-ui, sans-serif;
  --color-background: #99B2DD;
  --color-text: #00010;
  --color-primary: #832161;
  --color-secondary: #bdceea;
  --color-accent: #ADFC92;
  --color-warning: #df0000;
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  background-color: var(--color-background);
  font-family: var(--font-sans);
  color: var(--color-text);
  margin: 1rem;
}

h1, h2, h3 {
  color: var(--color-primary);
}

h2 {
  font-size: 1.2rem;
  display: block;
}

header {
}

header span {
  padding: .2rem;
  background-color: var(--color-primary);
  color: var(--color-secondary);
}

header .dot.on {
  color: var(--color-accent);
}
header .dot.off {
  color: var(--color-warning);
}

.control-box {
  padding: 1rem;
  border: 4px solid var(--color-primary);
  border-bottom: none;
  text-align: center;
}

.last-controls {
  border-bottom: 4px solid var(--color-primary);
}

.intro-text {
  font-size: 1.2rem;
  color: var(--color-primary);
}

.params span {
  color: var(--color-primary);
  padding: .2rem;
}

input {
  background-color: var(--color-secondary);
  border: none;
  padding: .4rem;
  font-family: var(--font-sans);
  color: var(--color-primary);
  font-weight: bold;
  text-align: center;
}

.prompt-input {
  width: 100%;
  height: 5rem;
  font-size: 1.2rem;
  border-bottom: 4px solid var(--color-primary);
}

.prompt-input::placeholder {
  color: var(--color-background);
  font-weight: normal;
}

button {
  background-color: var(--color-secondary);
  color: var(--color-primary);
  border: none;
  border-bottom: 4px solid var(--color-primary);
  font-family: var(--font-sans);
  padding: .4rem;
  margin: .4rem;
  font-size: 1.2rem;
  text-transform: uppercase;
}

button:hover {
  background-color: var(--color-accent);
  cursor: pointer;
}

.output {
  min-height: 5em;
  padding: 1rem;
  font-size: 3rem;
  line-height: 4rem;
}

.output span {
  margin: .2rem;
  color: var(--color-primary);
  background-color: var(--color-accent);
  transition: filter 0.35s ease;
  display: inline-block;
}

input::-webkit-outer-spin-button,
input::-webkit-inner-spin-button {
  -webkit-appearance: none;
  margin: 0;
}

/* Firefox */
input[type=number] {
  -moz-appearance: textfield;
}

.scales-header {
  font-weight: bold;
  color: var(--color-primary);
  margin-right: 0.5rem;
}

/* ── head selector row ────────────────────────────────────────────────── */
.head-row {
  display: flex;
  flex-wrap: wrap;
  width: 100%;
  margin-top: 0.5rem;
  gap: 2px;
}

.head-btn {
  flex: 1 1 auto;
  min-width: 2rem;
  font-size: 0.6rem;
  padding: 3px 2px;
  margin: 0;
  text-transform: none;
  border-bottom-width: 2px;
  opacity: 0.55;
  letter-spacing: 0;
}

.head-btn.head-active {
  background-color: var(--color-accent);
  opacity: 1;
  font-weight: 700;
}

.btn-row {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
}

.btn.stop {
  margin-left: auto;
  font-size: 1rem;
  padding: 0.4rem 0.7rem;
  color: var(--color-warning);
  border-bottom-color: var(--color-warning);
  opacity: 0.5;
}

.btn.stop:not(:disabled) {
  opacity: 1;
}

.btn.stop:not(:disabled):hover {
  background-color: var(--color-warning);
  color: var(--color-secondary);
}

.toggle {
  font-size: 1rem;
  padding: 0.3rem 0.8rem;
}

.toggle-active {
  background-color: var(--color-accent);
  border-bottom-color: var(--color-primary);
}

.tok-selected {
  outline: 3px solid var(--color-primary);
  outline-offset: -1px;
}

/* ── login overlay ────────────────────────────────────────────────────── */
.login-overlay {
  position: fixed;
  inset: 0;
  background: var(--color-background);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
}

.login-box {
  border: 4px solid var(--color-primary);
  padding: 2rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
  min-width: 280px;
  background: var(--color-background);
}

.login-box h2 {
  color: var(--color-primary);
  font-size: 1.4rem;
  text-align: center;
}

.login-box label {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
  color: var(--color-primary);
  font-weight: bold;
}

.login-box input {
  width: 100%;
  text-align: left;
}

.login-error {
  color: var(--color-warning);
  font-size: 0.9rem;
  text-align: center;
}

pre {
  display: none;
}

footer {
  border-top: 4px solid var(--color-primary);
  text-align: center;
}

footer img {
  margin-top: 2rem;
  max-width: 100px;
}
</style>
