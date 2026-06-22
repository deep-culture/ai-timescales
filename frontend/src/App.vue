<template>
  <div v-if="loginRequired && !loggedIn" class="login-overlay">
    <form class="login-box" @submit.prevent="doLogin">
      <h2>EigenzAIt</h2>
      <p v-if="loginError" class="login-error">{{ loginError }}</p>
      <label>Username<input v-model="loginUser" type="text" autocomplete="username" /></label>
      <label>Password<input v-model="loginPass" type="password" autocomplete="current-password" /></label>
      <button type="submit" :disabled="loginBusy">Sign in</button>
    </form>
  </div>

  <template v-else>
  <header>
    <span class="title"><strong>EigenzAIt</strong></span>
    <span :class="['dot', online ? 'on' : 'off']">●</span>
    <span class="status-text">{{ statusText }}</span>
  </header>

  <main>
    <div class="control-box intro-text">
      <div>How can we <a href="https://www.elgaronline.com/edcollchap/book/9781803928562/book-part-9781803928562-69.xml" target="_blank">critically listen</a> to AI temporalities at different timescales?</div>
      <div>What ‘gets through’ to <a href="https://www.tandfonline.com/doi/abs/10.1207/S15327884MCA0704_03" target="_blank">adjacent temporal strata</a>?</div>
      <div>What are the <a href="https://journals.sagepub.com/doi/10.1177/0263276413496286" target="_blank">distinct temporalities</a> of <a href="https://culturalanalytics.org/article/id/950/" target="_blank">different AI architectures</a>?</div>
    </div>
    <div class="control-box general-params">
      <div class="control-box-header">Generation settings</div>
      <span class="input">
        <label>Sequence length <input type="number" v-model.number="genLength" min="8" max="256" step="8" @change="snapConstraints()" /></label>
      </span>
      <span class="input">
        <label>Temperature <input type="number" v-model.number="temperature" min="0" max="2" step="0.1" /></label>
      </span>
    </div>
    <div class="control-box diffusion-params" v-if="diffusionModel">
      <div class="control-box-header">Diffusion settings</div>
      <span class="input">
        <label>Steps <input type="number" v-model.number="steps" min="1" max="256" step="1" @change="snapConstraints()" /></label>
      </span>
      <span class="input">
        <label>Block length <input type="number" v-model.number="blockLength" min="8" max="256" step="8" @change="snapConstraints()" /></label>
      </span>
      <span class="note">{{ constraintNote }}</span>
    </div>
    <div class="control-box diffusion-params" v-else>
      <div class="control-box-header">Diffusion settings</div>
      Load a diffusion model for diffusion settings
    </div>
    <!-- scales -->
    <div class="control-box scales">
      <div class="control-box-header">Timescale</div>
      <button :class="['btn', 'toggle', timescale === 'inference' ? 'toggle-active' : '']" @click="switchTimescale('inference')">Inference</button>
      <button :class="['btn', 'toggle', timescale === 'attention' ? 'toggle-active' : '']" @click="switchTimescale('attention')">Attention</button>
    </div>
    <!-- input + buttons -->
    <div class="control-box prompt" :class="timescale === 'inference' ? 'last-controls' : ''">
      <div class="control-box-header">Prompt</div>
      <div>
        <input class="prompt-input" v-model="userInput" placeholder="Insert prompt" @keydown.enter="onEnter" />
      </div>
      <div class="btn-row">
        <template v-if="timescale === 'inference'">
          <button class="btn ar" :disabled="busy || !online || !arModel" @click="runAR">Autoregress</button>
          <button class="btn diff" :disabled="busy || !online || !diffusionModel" @click="runDiffuse">Denoise</button>
          <div class="status-buttons">
            <button class="btn status" v-if="genStatus[0] && genStatus[1]" :class="genStatus[0]" :title="genStatus[1]">
              <span v-if="genStatus[0] == 'processing'" class="spinner"></span>
              <span v-else-if="genStatus[0] == 'warning'">⚠️</span>
              <span v-else-if="genStatus[0] == 'done'">✔</span>
            </button>
            <button class="btn stop" :disabled="!busy" @click="stopOrClear" title="Stop inference">&#9632;</button>
          </div>
        </template>
        <!-- attention: step the model one token/denoise at a time. Each button
             both runs its step and, once started, relabels to "next …". The
             status button on the right mirrors the inference status. -->
        <template v-else>
          <button
            class="btn ar" :disabled="busy || !online || !arModel"
            @click="nextARStep"
          >{{ arAttnTokens.length ? 'Next autoregression' : 'Autoregress' }}</button>
          <button
            class="btn diff" :disabled="busy || !online || !diffusionModel"
            @click="nextDiffStep"
          >{{ diffStepBuffer.length ? 'Next denoising' : 'Denoise' }}</button>
          <div class="status-buttons">
            <button class="btn status" v-if="genStatus[0] && genStatus[1]" :class="genStatus[0]" :title="genStatus[1]">
              <span v-if="genStatus[0] == 'processing'" class="spinner"></span>
              <span v-else-if="genStatus[0] == 'warning'">⚠️</span>
              <span v-else-if="genStatus[0] == 'done'">✔</span>
            </button>
            <button class="btn stop" :disabled="!busy" @click="stopOrClear" title="Stop step">&#9632;</button>
          </div>
        </template>
      </div>
    </div>
    <!-- head selector — only in attention timescale -->
    <div v-if="timescale === 'attention'" class="control-box">
      <div><h2>Attention heads</h2></div>
      <div class="attention-head-controls">
        <button
          :class="['head-btn', headAvg ? 'head-active' : '']"
          @click="toggleAvg()"
        >avg</button>
        <button
          v-for="h in nHeads" :key="h - 1"
          :class="['head-btn', (!headAvg && selectedHeads.has(h - 1)) ? 'head-active' : '']"
          @click="toggleHead(h - 1)"
        >{{ h - 1 }}</button>
        </div>
    </div>
    <!-- layer echo playback — plays all attention layers first→last -->
    <div v-if="timescale === 'attention'" class="control-box echo-row last-controls">
      <div><h2>Attention echoes</h2></div>
      <div class="btn-row">
        <button
          :class="['btn', 'toggle', playbackMode === 'eigenzeit' ? 'toggle-active' : '']"
          @click="playbackMode = 'eigenzeit'"
          title="Play layers at their real inter-layer delay (sub-millisecond)"
        >Eigenzeit</button>
        <button
          :class="['btn', 'toggle', playbackMode === 'interval' ? 'toggle-active' : '']"
          @click="playbackMode = 'interval'"
          title="Insert a 1-second gap between layers so each is audible"
        >Interval</button>
        <div class="status-buttons">
          <button
              class="btn status play"
              :disabled="!echoStep || echoPlaying"
              @click="playLayerEchoes"
              title="Play attention echoes"
            >
              <span v-if="echoPlaying" class="spinner"></span>
              <span v-else>▶︎</span>
            </button>
            <button class="btn stop" :disabled="!echoPlaying" @click="stopEchoesClicked" title="Stop attention echoes">&#9632;</button>
        </div>
      </div>
    </div>

    <div id="output-box">

      <!-- heartbeat graph -->
      <HeartbeatGraph
        v-if="timescale !== 'attention'"
        :points="heartbeat"
        :stepTimes="stepTimes"
        :graphTitle="graphTitle"
      />
      <!-- attention timescale: per-layer "heartbeat" on the Eigenzeit axis,
           with a dot marking the layer currently sonified -->
      <template v-else-if="attnPoints.length">
        <HeartbeatGraph
          :points="attnPoints"
          :stepTimes="attnLayerTimes"
          :activeIndex="echoLayer"
          :graphTitle="attnGraphLabel"
          :baseline="0"
          :isAttentionTimescale="true"
        />
        <div class="attn-graph-caption">
          <strong>Y = <a href="https://transformer-circuits.pub/2021/framework/index.html" target="_blank">Direct Logit Attribution</a></strong> (how much each attention layer
          pushes the final output toward the chosen token. The dashed line is zero:
          above it the layer promotes the token, below it the layer suppresses it).<br>
          <strong>X = Moment when each layer is reached</strong> (measured by CUDA events).<br>
          <strong>Dot = Currently playing.</strong>
        </div>
      </template>

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
    </div>
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

// ── types
// Per-layer "attention echoes" — present only on the final step of a sequence.
interface LayerData {
  n_layers: number
  n_heads: number
  timings_ns: number[]                          // per-layer GPU-execution Eigenzeit (ns offsets, layer 0 = 0)
  diffusion: boolean
  attention: number[][] | number[][][]          // per-layer mean: (L,T) AR or (L,gen,gen) DLM
  attention_heads: number[][][] | number[][][][] // per-layer per-head: (L,H,T) or (L,H,gen,gen)
  dla?: number[] | number[][]                   // direct logit attribution: (L,) AR or (L,gen) DLM
}

interface StepEvent {
  step_index: number
  elapsed_s: number               // seconds since generation start (backend clock)
  token_ids: number[]
  decoded_tokens: string[]
  mask_positions: number[]
  newly_revealed: number[]
  attention?: number[] | number[][]          // mean: 1D (AR) or 2D (diffusion)
  attention_heads?: number[][] | number[][][] // per-head: (n_heads,T) or (n_heads,gen,T)
  n_heads?: number
  prompt_length?: number
  prompt_tokens?: string[]        // decoded prompt + template tokens (attention timescale)
  prompt_specials?: boolean[]     // per prompt token: is it a special/template token?
  decoded_specials?: boolean[]    // per generated token: is it a special token?
  layers?: LayerData
}

interface HeartbeatPoint { x: number; y: number }

// ── auth
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

// ── state
const online = ref(false)
const busy = ref(false)
const statusText = ref('Connecting…')
const genStatus =  ref<[string, string]>(['', ''])
const userInput = ref('')

const arModel = ref<string | null>(null)
const diffusionModel = ref<string | null>(null)

const tokens = reactive<{ text: string; cls: string }[]>([])
const heartbeat = reactive<HeartbeatPoint[]>([])
const stepTimes = reactive<number[]>([])
let t0 = 0

// ── model badge
const _tokensPerSecHistory = reactive<number[]>([])
let _lastStepTime = 0
const _currentRunModel = ref<string | null>(null)
const _currentRunType = ref<'AR' | 'diffusion' | null>(null)

const graphTitle = computed(() => {
  const m = _currentRunModel.value
  if (!m) return ''
  const short = m.split('/').pop() ?? m
  const type = _currentRunType.value ?? ''
  const avg = _tokensPerSecHistory.length
    ? (_tokensPerSecHistory.reduce((a, b) => a + b, 0) / _tokensPerSecHistory.length).toFixed(1)
    : null
  return avg ? `${short} · ~${avg} tokens per second` : `${short}`
})

function resetRunMetrics(model: string, type: 'AR' | 'diffusion') {
  _currentRunModel.value = model
  _currentRunType.value = type
  _tokensPerSecHistory.length = 0
  _lastStepTime = 0
}

function recordStepTempo(newTokenCount: number, elapsed_s: number) {
  if (_lastStepTime > 0 && newTokenCount > 0) {
    const dt = elapsed_s - _lastStepTime
    if (dt > 0) _tokensPerSecHistory.push(newTokenCount / dt)
  }
  _lastStepTime = elapsed_s
  stepTimes.push(elapsed_s)
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

// ── timescale mode
const timescale = ref<'inference' | 'attention'>('inference')
const selectedTokenIdx = ref(0)

// ── head selection (multi-select)
const headAvg = ref(true)
const selectedHeads = reactive(new Set<number>())
const nHeads = ref(32)
const availableVoices = ref<string[]>(['af_heart'])
const tokenBlurs = reactive<number[]>([])

// ── layer "echo" playback
// Layer data from the final step of the last sequence (carries every layer's
// attention + per-layer timings). One voice per layer, played first→last.
const echoStep = ref<StepEvent | null>(null)
// 'eigenzeit' → schedule each layer at its real (sub-ms) inter-layer delay;
// 'interval'  → insert a 2-second gap between layers so it's audible.
const playbackMode = ref<'eigenzeit' | 'interval'>('interval')
const echoPlaying = ref(false)
const echoLayer = ref(-1)              // layer currently sounding/blurring (-1 = idle)
// Echo playback status (spinner/done/warning) — shown in the echo-row, kept
// separate from genStatus so it never appears in the prompt status button.
const echoStatus = ref<[string, string]>(['', ''])
let _echoTimers: number[] = []         // pending blur-update timers, cleared on stop

// Fixed voice palette: index 0 = avg voice, indices 1-4 = head voices (cycling)
const DEFAULT_VOICES = ['af_heart', 'af_nicole', 'am_michael', 'bf_alice', 'bm_fable']

// ── derived state
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
    updateARBlurs(lastStep.value)
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

const MAX_BLUR = 8

// ── full-sequence attention display model
// In the attention timescale the displayed token stream is the WHOLE sequence
// the model attends over: prompt + chat-template/special tokens, then the
// response. The invariant that keeps the bookkeeping sane:
//     display index  ==  sequence position  ==  attention key (column) index.
// Prompt tokens are shown and blur with attention, but are never the query
// (AR query = last token; DLM query = a clicked response token) and never
// drive TTS if they are special/template tokens.

// Number of prompt (incl. template) tokens carried by an attention step/echo.
// Zero outside the attention timescale (the backend only sends them there).
function promptLen(ev: StepEvent | null | undefined): number {
  return ev?.prompt_tokens?.length ?? 0
}

// Is the token at display index d a special/template token?
function isSpecialAt(ev: StepEvent, d: number): boolean {
  const PL = promptLen(ev)
  return d < PL ? !!ev.prompt_specials?.[d] : !!ev.decoded_specials?.[d - PL]
}

// Append the prompt + template tokens to the `tokens` array (no-op when none).
function pushPromptTokens(ev: StepEvent) {
  const pt = ev.prompt_tokens
  if (!pt) return
  for (let i = 0; i < pt.length; i++) {
    tokens.push({
      text: pt[i] || ' ',
      cls: ev.prompt_specials?.[i] ? 'tok-special' : 'tok-prompt',
    })
  }
}

// CSS class for a response token at response-index r.
function responseClass(ev: StepEvent, r: number, revealed: boolean): string {
  if (ev.decoded_specials?.[r]) return 'tok-special'
  return revealed ? 'tok-new' : 'tok-old'
}

// Keep the DLM selection on a real response token (prompt tokens aren't
// focusable). Display index PL = first response token.
function clampDiffSelection(ev: StepEvent) {
  const PL = promptLen(ev)
  const T = PL + ev.decoded_tokens.length
  if (selectedTokenIdx.value < PL) selectedTokenIdx.value = PL
  if (selectedTokenIdx.value > T - 1) selectedTokenIdx.value = Math.max(PL, T - 1)
}

function updateARBlurs(ev: StepEvent) {
  const row = getEffectiveAttnRow(ev)          // full-T last-row (key per position)
  const T = promptLen(ev) + arAttnTokens.length
  const q = T - 1                              // AR query = last token
  const maxS = Math.max(...row.slice(0, T).filter((_, d) => d !== q), 1e-9)
  tokenBlurs.length = 0
  for (let d = 0; d < T; d++) {
    tokenBlurs.push(d === q ? 0 : MAX_BLUR * (1 - (row[d] ?? 0) / maxS))
  }
}

function updateDiffusionBlurs(ev: StepEvent) {
  clampDiffSelection(ev)
  const PL = promptLen(ev)
  const T = PL + ev.decoded_tokens.length
  const q = selectedTokenIdx.value             // display index of the query
  const maskSet = new Set(ev.mask_positions.map(m => PL + m))
  const row = getEffectiveAttnRow(ev, q - PL)  // row is gen-indexed → strip prompt
  const nonMaskMax = Math.max(...row.slice(0, T).filter((_, d) => !maskSet.has(d)), 1e-9)
  tokenBlurs.length = 0
  for (let d = 0; d < T; d++) {
    if (d === q) { tokenBlurs.push(0); continue }
    if (maskSet.has(d)) { tokenBlurs.push(MAX_BLUR); continue }
    tokenBlurs.push(MAX_BLUR * (1 - (row[d] ?? 0) / nonMaskMax))
  }
}

// Blur the tokens by ONE layer's attention (used while echoes play, so the
// visual sweeps layer-by-layer in step with the audio). Mirrors the AR/DLM
// blur maths above but reads the per-layer row instead of the merged step row.
function applyLayerBlur(layers: LayerData, l: number) {
  const ev = echoStep.value
  if (!ev) return
  const PL = promptLen(ev)
  if (!layers.diffusion) {
    const T = PL + arAttnTokens.length
    const row = getLayerRow(layers, l)         // full-T last row
    const q = T - 1
    const maxS = Math.max(...row.slice(0, T).filter((_, d) => d !== q), 1e-9)
    tokenBlurs.length = 0
    for (let d = 0; d < T; d++) {
      tokenBlurs.push(d === q ? 0 : MAX_BLUR * (1 - (row[d] ?? 0) / maxS))
    }
  } else {
    const T = PL + ev.decoded_tokens.length
    const q = selectedTokenIdx.value
    const maskSet = new Set(ev.mask_positions.map(m => PL + m))
    const row = getLayerRow(layers, l, q - PL)  // row is gen-indexed
    const nonMaskMax = Math.max(...row.slice(0, T).filter((_, d) => !maskSet.has(d)), 1e-9)
    tokenBlurs.length = 0
    for (let d = 0; d < T; d++) {
      if (d === q) { tokenBlurs.push(0); continue }
      if (maskSet.has(d)) { tokenBlurs.push(MAX_BLUR); continue }
      tokenBlurs.push(MAX_BLUR * (1 - (row[d] ?? 0) / nonMaskMax))
    }
  }
}

// AR attention mode state
const arAttnTokens = reactive<string[]>([])   // accumulated generated token texts
const arAttnSpecials = reactive<boolean[]>([]) // parallel: is each token special?
// The original prompt (incl. template) captured on the FIRST AR step. On
// continue_only steps the backend's "prompt" grows to include generated tokens,
// so we pin the prompt prefix to this to avoid duplicating the response.
let arPromptTokens: string[] = []
let arPromptSpecials: boolean[] = []
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

// ── heartbeat SVG path (reactive)

// ── snap constraints
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

// ── backend polling
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

// SSE streaming
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
      genStatus.value = ['warning', `Server error ${res.status}`]
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
          else if (eventType === 'cancelled') {
            genStatus.value = ['', ''];
            return { ok: false, finalIds: [], finalText: ''
            }
          }
          else if (eventType === 'error') {
            genStatus.value = ['warning', `${data.message}`];
            return { ok: false, finalIds: [], finalText: ''
            }
          }
        }
      }
    }
    return { ok: true, finalIds, finalText }
  } catch (e: any) {
    if (e.name === 'AbortError') {
      genStatus.value = ['warning', 'Stopped.']; return { ok: false, finalIds: [], finalText: '' } }
      genStatus.value = ['warning', `${e.message}`]
      return { ok: false, finalIds: [], finalText: '' }
  } finally {
    abortCtrl = null
  }
}

// ── TTS + display queue (Option C: step buffering)
// Generation runs freely; TTS fetches fire in parallel; display waits per-step
// until its audio is ready, then plays + renders simultaneously.

const SKIP_TOKENS = new Set(['<|endoftext|>', '<|eot_id|>', '<|end_header_id|>', '<eos>', '<s>', '</s>', '<pad>'])

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


// timescale switching
function switchTimescale(mode: 'inference' | 'attention') {
  timescale.value = mode
  // Reset state when switching
  tokens.length = 0
  heartbeat.length = 0;
  stepTimes.length = 0
  lastStep.value = null
  genStatus.value = ['', '']
  resetDisplayQueue()
  // Reset attention-mode state
  arAttnTokens.length = 0
  arAttnSpecials.length = 0
  arPromptTokens = []
  arPromptSpecials = []
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
  echoStep.value = null
  stopEchoes()
  echoStatus.value = ['', '']
}

// pan helper
function panForIndex(i: number, total: number): number {
  const t = total > 1 ? i / (total - 1) : 0.5
  return (t * 2) - 1
}

// ── volume-aware TTS playback
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

// ── layer "echo" playback
// Plays the per-layer attention scores layer-by-layer (first→last). Each layer
// gets its own TTS voice; a token's amplitude scales with its attention score
// (loud = attended, whisper = ignored). MASK tokens (diffusion) become white
// noise. Tokens are panned left→right by their position in the sequence.

let _noiseBuffer: AudioBuffer | null = null
function getNoiseBuffer(ctx: AudioContext, dur = 0.35): AudioBuffer {
  if (_noiseBuffer && _noiseBuffer.sampleRate === ctx.sampleRate) return _noiseBuffer
  const len = Math.floor(ctx.sampleRate * dur)
  const buf = ctx.createBuffer(1, len, ctx.sampleRate)
  const d = buf.getChannelData(0)
  for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * 0.4
  _noiseBuffer = buf
  return buf
}

// Sources scheduled by the echo playback, kept so a stop can cancel them.
let _echoSources: AudioBufferSourceNode[] = []

// Schedule already-decoded buffers at an absolute AudioContext time.
function playAudioBuffersAt(items: { buffer: AudioBuffer; pan: number; volume: number }[], startTime: number): void {
  if (!items.length) return
  const ctx = getAudioCtx()
  for (const { buffer, pan, volume } of items) {
    try {
      const src = ctx.createBufferSource()
      src.buffer = buffer
      const gainNode = ctx.createGain()
      gainNode.gain.value = volume * ttsVolume.value
      const panner = ctx.createStereoPanner()
      panner.pan.value = pan
      src.connect(panner); panner.connect(gainNode); gainNode.connect(ctx.destination)
      src.start(startTime)
      _echoSources.push(src)
    } catch { /* best-effort */ }
  }
}

// Schedule a burst of white-noise "tokens" (masks) at an absolute time.
function playNoiseAt(noise: { pan: number; volume: number }[], startTime: number): void {
  if (!noise.length) return
  const ctx = getAudioCtx()
  const buf = getNoiseBuffer(ctx)
  for (const { pan, volume } of noise) {
    try {
      const src = ctx.createBufferSource()
      src.buffer = buf
      const gainNode = ctx.createGain()
      gainNode.gain.value = volume * ttsVolume.value
      const panner = ctx.createStereoPanner()
      panner.pan.value = pan
      src.connect(panner); panner.connect(gainNode); gainNode.connect(ctx.destination)
      src.start(startTime)
      _echoSources.push(src)
    } catch { /* best-effort */ }
  }
}

// Cancel any audio the echo playback has scheduled but not yet finished.
function stopEchoSources(): void {
  for (const src of _echoSources) {
    try { src.stop() } catch { /* already stopped/ended */ }
  }
  _echoSources = []
}

// Token position → voice (cycles through the palette when tokens outnumber it).
// Each token keeps the same voice across every layer, so a token's identity is
// its voice while its sequence position is its pan.
function tokenVoice(i: number): string {
  const palette = availableVoices.value.length ? availableVoices.value : DEFAULT_VOICES
  return palette[i % palette.length]
}

// Per-layer attention row, averaged over the active head selection.
// rowIdx: undefined = AR (full last row); number = DLM (selected-token row).
function getLayerRow(layers: LayerData, l: number, rowIdx?: number): number[] {
  const useMean = headAvg.value || selectedHeads.size === 0
  const isDiff = rowIdx !== undefined
  if (useMean) {
    const a = layers.attention as any
    return (isDiff ? a[l]?.[rowIdx!] : a[l]) ?? []
  }
  const ah = layers.attention_heads as any
  const heads = [...selectedHeads]
  const rows: number[][] = heads.map(h => (isDiff ? ah[l]?.[h]?.[rowIdx!] : ah[l]?.[h]) ?? [])
  if (!rows.length || !rows[0].length) {
    const a = layers.attention as any
    return (isDiff ? a[l]?.[rowIdx!] : a[l]) ?? []
  }
  const len = rows[0].length
  return Array.from({ length: len }, (_, i) =>
    rows.reduce((s, r) => s + (r[i] ?? 0), 0) / rows.length)
}

// Build one layer's playback: attention-weighted TTS tokens + noise (masks).
// Iterates the FULL sequence (prompt + response): prompt words are spoken at
// their attention weight, special/template tokens are muted, masks → noise, and
// each token's voice follows its sequence position.
function buildLayerSchedule(ev: StepEvent, layers: LayerData, l: number):
  { texts: string[]; pans: number[]; vols: number[]; voices: string[]; noise: { pan: number; volume: number }[] } {
  const PL = promptLen(ev)
  const isDiff = layers.diffusion
  const respTexts = isDiff ? ev.decoded_tokens : (arAttnTokens as string[])
  const T = PL + respTexts.length
  const q = isDiff ? selectedTokenIdx.value : T - 1   // query display index
  const row = getLayerRow(layers, l, isDiff ? q - PL : undefined)
  const maskSet = new Set(ev.mask_positions.map(m => PL + m))
  const norm = Math.max(...row.slice(0, T).filter((_, d) => d !== q && !maskSet.has(d)), 1e-9)

  const texts: string[] = [], pans: number[] = [], vols: number[] = [], voices: string[] = []
  const noise: { pan: number; volume: number }[] = []
  for (let d = 0; d < T; d++) {
    if (d === q) continue
    const pan = panForIndex(d, T)
    const v = (row[d] ?? 0) / norm
    if (maskSet.has(d)) { noise.push({ pan, volume: Math.min(1, v || 0.3) }); continue }
    if (isSpecialAt(ev, d)) continue            // mute special/template tokens
    const text = d < PL ? (ev.prompt_tokens?.[d] ?? '') : respTexts[d - PL]
    if (!text || !text.trim()) continue
    texts.push(text); pans.push(pan); vols.push(v); voices.push(tokenVoice(d))
  }
  return { texts, pans, vols, voices, noise }
}

// Per-layer start offsets (seconds, relative to the first layer).
function computeLayerOffsets(layers: LayerData): number[] {
  const L = layers.n_layers
  if (playbackMode.value === 'interval') {
    return Array.from({ length: L }, (_, l) => l * 1.0)   // 1-second audible gap
  }
  const t = layers.timings_ns
  if (!t?.length) return Array.from({ length: L }, (_, l) => l * 0.001)
  const t0 = t[0]
  return t.map(x => Math.max(0, x - t0) / 1e9)             // real Eigenzeit (ns → s)
}

// Direct logit attribution of layer l toward the chosen token. AR = one value
// per layer; DLM = one value per (layer, generated position) → pick the selected
// token's column. Falls back to the layer index if no DLA was sent.
function layerDla(layers: LayerData, l: number): number {
  const dla = layers.dla
  if (!dla) return l
  // DLM dla is per generated position → strip the prompt offset from the
  // display-indexed selection to get the response column.
  if (layers.diffusion) {
    const g = selectedTokenIdx.value - promptLen(echoStep.value)
    return (dla as number[][])[l]?.[Math.max(0, g)] ?? 0
  }
  return (dla as number[])[l] ?? 0
}

// Attention "heartbeat": one point per layer. x = Eigenzeit (always — even when
// the 2s interval is chosen for playback), y = that layer's Direct Logit
// Attribution, i.e. how much the layer's attention pushes the final output
// toward the chosen token (signed: positive = promotes it, negative = suppresses).
const attnPoints = computed<{ x: number; y: number }[]>(() => {
  const layers = echoStep.value?.layers
  if (!layers) return []
  const L = layers.n_layers
  const t = layers.timings_ns
  const t0 = (t && t.length) ? t[0] : 0
  return Array.from({ length: L }, (_, l) => ({
    x: (t && t.length) ? Math.max(0, t[l] - t0) / 1e9 : l,
    y: layerDla(layers, l),
  }))
})

// Per-layer Eigenzeit timestamps (seconds) → tick marks under the curve.
const attnLayerTimes = computed<number[]>(() => {
  const t = echoStep.value?.layers?.timings_ns
  if (!t?.length) return []
  const t0 = t[0]
  return t.map(x => Math.max(0, x - t0) / 1e9)
})

const attnGraphLabel = computed(() => {
  const layers = echoStep.value?.layers
  if (!layers) return ''
  let modelName = _currentRunModel.value
  if (!modelName) return ''
  modelName = modelName.split('/').pop() ?? modelName
  return `${modelName} · ${layers.n_layers} layers`
})

let _echoWaitTimer: number | undefined
let _echoWaitResolve: (() => void) | null = null

function clearEchoTimers() {
  for (const t of _echoTimers) clearTimeout(t)
  _echoTimers = []
}

// Stop an in-flight echo playback: cancel pending blur sweeps + scheduled audio
// and release the wait so playLayerEchoes can unwind.
function stopEchoes() {
  clearEchoTimers()
  stopEchoSources()
  if (_echoWaitTimer !== undefined) { clearTimeout(_echoWaitTimer); _echoWaitTimer = undefined }
  echoLayer.value = -1
  if (_echoWaitResolve) { const r = _echoWaitResolve; _echoWaitResolve = null; r() }
}

// Stop button handler (prompt box): halt playback and report it in the echo-row.
function stopEchoesClicked() {
  stopEchoes()
  echoStatus.value = ['warning', 'Echoes stopped']
}

async function playLayerEchoes() {
  const ev = echoStep.value
  if (!ev?.layers) { echoStatus.value = ['warning', 'Run a step in attention mode first']; return }
  if (echoPlaying.value) return
  echoPlaying.value = true
  clearEchoTimers()
  _echoSources = []
  const layers = ev.layers
  const L = layers.n_layers

  // Show the final-state tokens so the per-layer blur maps onto what's on screen.
  if (layers.diffusion) { renderDiffusion(ev); clampDiffSelection(ev); lastStep.value = ev }

  echoStatus.value = ['processing', `Playing ${L} attention layers (${playbackMode.value})…`]
  try {
    // Pre-fetch every layer's TTS buffers in parallel so scheduling is precise.
    const scheds = Array.from({ length: L }, (_, l) => buildLayerSchedule(ev, layers, l))
    const bufsPerLayer = await Promise.all(scheds.map(s =>
      fetchTtsBuffersWithVolumes(s.texts, s.pans, s.vols, s.voices)))

    const ctx = getAudioCtx()
    const offsets = computeLayerOffsets(layers)
    const leadS = 0.15
    const startAt = ctx.currentTime + leadS
    for (let l = 0; l < L; l++) {
      const at = startAt + offsets[l]
      playAudioBuffersAt(bufsPerLayer[l], at)
      playNoiseAt(scheds[l].noise, at)
      // Sweep the blur to this layer in step with its audio (same lead/offset).
      _echoTimers.push(window.setTimeout(() => {
        echoLayer.value = l
        applyLayerBlur(layers, l)
        echoStatus.value = ['processing', `layer ${l + 1}/${L}`]
      }, (leadS + offsets[l]) * 1000))
    }
    const totalDur = (offsets[L - 1] ?? 0) + leadS + 1.2
    await new Promise<void>(resolve => {
      _echoWaitResolve = resolve
      _echoWaitTimer = window.setTimeout(resolve, totalDur * 1000)
    })
    if (echoLayer.value !== -1) echoStatus.value = ['done', 'Echoes done']
  } finally {
    echoLayer.value = -1
    _echoWaitResolve = null
    _echoWaitTimer = undefined
    echoPlaying.value = false
  }
}

// ── single-step fetch helper
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

// ── AR attention mode
async function nextARStep() {
  if (busy.value || !online.value || !arModel.value) return
  const msg = userInput.value.trim()
  if (!msg && arAttnIds.value.length === 0) return
  busy.value = true
  genStatus.value = ['processing', 'Next AR token…']

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

  if (!stepEvts.length) { busy.value = false; genStatus.value = ['warning', 'No step']; return }
  const ev = stepEvts[0]
  arAttnIds.value = finalIds
  const newText = ev.decoded_tokens[0] ?? ''
  arAttnTokens.push(newText)
  arAttnSpecials.push(ev.decoded_specials?.[0] ?? false)
  // Pin the prompt prefix to the original prompt, and expose the *accumulated*
  // response specials, so every full-sequence helper sees correct labels.
  if (isFirst) {
    arPromptTokens = ev.prompt_tokens ?? []
    arPromptSpecials = ev.prompt_specials ?? []
  }
  ev.prompt_tokens = arPromptTokens
  ev.prompt_specials = arPromptSpecials
  ev.decoded_specials = arAttnSpecials as boolean[]
  arLastAttnRow.value = (ev.attention as number[]) ?? null
  lastStep.value = ev
  if (ev.layers) echoStep.value = ev   // final step carries per-layer data

  // Render the full attended sequence: prompt + template tokens, then the
  // accumulated response (last token = query, always sharp).
  const PL = promptLen(ev)
  tokens.length = 0
  pushPromptTokens(ev)
  for (let i = 0; i < arAttnTokens.length; i++) {
    tokens.push({
      text: arAttnTokens[i] || '\u00a0',
      cls: i === arAttnTokens.length - 1
        ? 'tok-new'
        : (arAttnSpecials[i] ? 'tok-special' : 'tok-old'),
    })
  }
  const T = PL + arAttnTokens.length
  // Update blur based on current head selection
  updateARBlurs(ev)
  scrollOutput()

  // TTS: play new token first at full volume (always sharp / selected),
  // in its own per-position voice.
  const newBufs = await fetchTtsBuffersWithVolumes(
    [newText], [panForIndex(T - 1, T)], [1], [tokenVoice(T - 1)])
  let dur = 0
  if (newBufs.length) {
    dur = playAudioBuffersWithVolumes(newBufs)
    await new Promise(r => setTimeout(r, dur * 1000 + 80))
  }

  // Replay the rest of the sequence — prompt words + previous response tokens
  // (special/template tokens muted). Each token keeps its own per-position voice;
  // its volume is the attention weight (averaged over the active head selection).
  const q = T - 1
  const speakIdx: number[] = []
  for (let d = 0; d < T; d++) {
    if (d === q || isSpecialAt(ev, d)) continue
    const text = d < PL ? (ev.prompt_tokens?.[d] ?? '') : arAttnTokens[d - PL]
    if (text && text.trim()) speakIdx.push(d)
  }
  if (speakIdx.length) {
    const texts = speakIdx.map(d => d < PL ? (ev.prompt_tokens?.[d] ?? '') : arAttnTokens[d - PL])
    const pans = speakIdx.map(d => panForIndex(d, T))
    const voices = speakIdx.map(d => tokenVoice(d))
    const row = getEffectiveAttnRow(ev)
    const raw = speakIdx.map(d => row[d] ?? 0)
    const maxV = Math.max(...raw, 1e-9)
    const normVols = raw.map(v => v / maxV)
    const bufs = await fetchTtsBuffersWithVolumes(texts, pans, normVols, voices)
    playAudioBuffersWithVolumes(bufs)
  }

  genStatus.value = ['done', `Token ${arAttnTokens.length}`]
  busy.value = false
}

// ── Diffusion attention mode
async function nextDiffStep() {
  if (busy.value || !online.value || !diffusionModel.value) return

  // If buffer is exhausted or empty, start a new generation
  if (diffStepIdx.value >= diffStepBuffer.length) {
    const msg = userInput.value.trim()
    if (!msg) return
    busy.value = true
    genStatus.value = ['processing', 'Generating all steps…']
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
    // The final step carries every layer's attention — use it for echo playback.
    echoStep.value = [...allSteps].reverse().find(s => s.layers) ?? null

    if (!diffStepBuffer.length) { busy.value = false; genStatus.value = ['warning', 'No steps']; return }
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

  genStatus.value = ['done', `step ${diffStepIdx.value}/${diffStepBuffer.length}`]
  busy.value = false
}

async function playDiffusionAttentionTTS(ev: StepEvent) {
  clampDiffSelection(ev)
  const PL = promptLen(ev)
  const T = PL + ev.decoded_tokens.length
  const sel = selectedTokenIdx.value               // display index of the query
  const maskSet = new Set(ev.mask_positions.map(m => PL + m))

  // Selected token: play at full volume in its own per-position voice (always sharp)
  if (!maskSet.has(sel)) {
    const selText = sel < PL ? (ev.prompt_tokens?.[sel] ?? '') : ev.decoded_tokens[sel - PL]
    const selBufs = await fetchTtsBuffersWithVolumes(
      [selText], [panForIndex(sel, T)], [1], [tokenVoice(sel)])
    if (selBufs.length) {
      const dur = playAudioBuffersWithVolumes(selBufs)
      await new Promise(r => setTimeout(r, dur * 1000 + 80))
    }
  }

  // Other tokens across the FULL sequence: prompt words + response tokens,
  // muting special/template tokens (masks are handled as noise during echoes).
  const otherIdxs: number[] = [], otherTexts: string[] = [], otherPans: number[] = []
  for (let d = 0; d < T; d++) {
    if (d === sel || maskSet.has(d) || isSpecialAt(ev, d)) continue
    const text = d < PL ? (ev.prompt_tokens?.[d] ?? '') : ev.decoded_tokens[d - PL]
    if (!text || !text.trim()) continue
    otherIdxs.push(d)
    otherTexts.push(text)
    otherPans.push(panForIndex(d, T))
  }
  if (!otherTexts.length) return

  // Each token keeps its own per-position voice; its volume is the attention
  // weight toward the selected query (averaged over the active head selection).
  const row = getEffectiveAttnRow(ev, sel - PL)   // row is gen-indexed → strip prompt
  const voices = otherIdxs.map(d => tokenVoice(d))
  const rawVols = otherIdxs.map(d => row[d] ?? 0)
  const maxV = Math.max(...rawVols, 1e-9)
  const normVols = rawVols.map(v => v / maxV)
  const bufs = await fetchTtsBuffersWithVolumes(otherTexts, otherPans, normVols, voices)
  playAudioBuffersWithVolumes(bufs)
}

// ── token click handler
function onTokenClick(i: number) {
  // Only interactive for diffusion attention mode
  if (!isDiffAttnMode.value) return
  // Prompt + template tokens are shown but not focusable (they're never queries).
  if (i < promptLen(lastStep.value)) return
  selectedTokenIdx.value = i
  if (lastStep.value) {
    updateDiffusionBlurs(lastStep.value)
    playDiffusionAttentionTTS(lastStep.value)
  }
}

// ── token rendering helpers
function renderAR(ev: StepEvent) {
  const revSet = new Set(ev.newly_revealed)
  tokens.length = 0
  pushPromptTokens(ev)   // attention timescale only (no-op otherwise)
  for (let i = 0; i < ev.decoded_tokens.length; i++) {
    tokens.push({ text: ev.decoded_tokens[i] || '\u00a0', cls: responseClass(ev, i, revSet.has(i)) })
  }
}

function renderDiffusion(ev: StepEvent) {
  const maskSet = new Set(ev.mask_positions)
  const revSet = new Set(ev.newly_revealed)
  tokens.length = 0
  pushPromptTokens(ev)   // attention timescale only (no-op otherwise)
  for (let i = 0; i < ev.decoded_tokens.length; i++) {
    if (maskSet.has(i)) tokens.push({ text: '▒', cls: 'tok-mask' })
    else tokens.push({ text: ev.decoded_tokens[i] || '\u00a0', cls: responseClass(ev, i, revSet.has(i)) })
  }
}

function pushHeartbeat(ev: StepEvent) {
  const elapsed = ev.elapsed_s
  const unmasked = ev.decoded_tokens.length - ev.mask_positions.length
  heartbeat.push({ x: elapsed, y: unmasked })
  recordStepTempo(ev.newly_revealed.length, elapsed)
  if (ev.n_heads) nHeads.value = ev.n_heads
  lastStep.value = ev
}

// ── actions
const outputEl = ref<HTMLElement | null>(null)

function scrollOutput() {
  nextTick(() => { if (outputEl.value) outputEl.value.scrollTop = outputEl.value.scrollHeight })
}

async function runAR() {
  if (busy.value || !online.value || !arModel.value) return
  const msg = userInput.value.trim(); if (!msg) return
  busy.value = true
  genStatus.value = ['processing', `Generating with ${arModel.value}`]
  tokens.length = 0
  heartbeat.length = 0;
  stepTimes.length = 0
  lastStep.value = null
  t0 = performance.now()
  resetDisplayQueue()
  resetRunMetrics(arModel.value, 'AR')
  // Never retain context between inference-mode runs — avoids growing GPU allocations
  prevIds.ar = []

  const { ok } = await streamGenerate(arModel.value, msg, [], (ev) => {
    enqueueStep(ev, renderAR)
  })

  genStatus.value = ok ? ['done', 'Done'] : ['warning', genStatus.value[1]]
  busy.value = false
}

async function runDiffuse() {
  if (busy.value || !online.value || !diffusionModel.value) return
  const msg = userInput.value.trim(); if (!msg) return
  busy.value = true
  genStatus.value = ['processing', `${diffusionModel.value}…`]
  tokens.length = 0
  heartbeat.length = 0;
  stepTimes.length = 0
  lastStep.value = null
  t0 = performance.now()
  resetDisplayQueue()
  resetRunMetrics(diffusionModel.value, 'diffusion')
  // Never retain context between inference-mode runs — avoids growing GPU allocations
  prevIds.diffusion = []

  const { ok } = await streamGenerate(diffusionModel.value, msg, [], (ev) => {
    enqueueStep(ev, renderDiffusion)
  })
  genStatus.value = ok ? ['done', 'Done'] : ['warning', genStatus.value[1]]
  busy.value = false
}

async function runBoth() {
  if (busy.value || !online.value) return
  const msg = userInput.value.trim(); if (!msg) return
  busy.value = true
  tokens.length = 0
  heartbeat.length = 0;
  stepTimes.length = 0
  lastStep.value = null
  t0 = performance.now()
  resetDisplayQueue()
  prevIds.ar = []; prevIds.diffusion = []

  if (arModel.value) {
    genStatus.value = ['processing', `${arModel.value}…`]
    resetRunMetrics(arModel.value, 'AR')
    const { ok } = await streamGenerate(arModel.value, msg, [], (ev) => {
      enqueueStep(ev, renderAR)
    })
    if (!ok) { busy.value = false; return }
  }

  // reset for diffusion pass
  tokens.length = 0;
  heartbeat.length = 0;
  stepTimes.length = 0;
  lastStep.value = null;
  t0 = performance.now()
  resetDisplayQueue()

  if (diffusionModel.value) {
    genStatus.value = ['processing', `${diffusionModel.value}…`]
    resetRunMetrics(diffusionModel.value, 'diffusion')
    const { ok } = await streamGenerate(diffusionModel.value, msg, [], (ev) => {
      enqueueStep(ev, renderDiffusion)
    })
    genStatus.value = ok ? ['done', 'Done'] : ['warning', genStatus.value[1]]
  }
  busy.value = false
}

async function stopOrClear() {
  stopEchoes()  // also halt any running attention-echo playback
  if (busy.value) {
    abortCtrl?.abort()
    resetDisplayQueue()
    try { await fetch('/api/stop', { method: 'POST', headers: authHeaders() }) } catch {}
  } else {
    tokens.length = 0
    heartbeat.length = 0;
    stepTimes.length = 0
    lastStep.value = null
    genStatus.value = ['', '']
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
  --color-warning: #ff9900;
  --color-error: #df0000;
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

a:not(:has(img)) {
  color: var(--color-primary);
  text-decoration: none;
  border-bottom: 4px solid;
}

a:hover {
  background-color: var(--color-accent);
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
  color: var(--color-error);
}

.control-box {
  padding: 1rem;
  border: 4px solid var(--color-primary);
  border-bottom: none;
  text-align: center;
}

.control-box-header {
  color: var(--color-primary);
  margin: 0;
  padding: 0;
  margin-bottom: .5rem;
  font-weight: bold;
}

.control-box > .input {
  margin-right: .5rem;
}

.last-controls {
  border-bottom: 4px solid var(--color-primary);
}

.intro-text {
  font-size: 1.2rem;
}

.intro-text > div {
  margin-bottom: .5rem;
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

button:disabled {
  color: var(--color-background);
  cursor: not-allowed;
}

button:not(:disabled):hover {
  background-color: var(--color-accent);
  cursor: pointer;
}


#output-box {
  margin-top: 1rem;
}

.output {
  min-height: 5em;
  margin-top: 1rem;
  font-size: 2rem;
  line-height: 2rem;
}

.output span {
  margin: .2rem;
  padding: .3rem;
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

/* head selector row */
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

/* layer echo playback row */
.box-row {
  width: 100%;
  margin-top: 0.75rem;
}

.attention-head-controls, .echo-controls {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
  margin-top: .5rem;
}

.echo-controls {
  justify-content: right;
}

.attn-graph-caption {
  font-size: 0.7rem;
  color: var(--color-primary);
  opacity: 0.65;
  text-align: left;
  margin-top: 1rem;
  line-height: 1.4;
}

.attn-graph-caption a {
  border-bottom: 2px solid;
}

.btn-row {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
}

.status-buttons {
  margin-left: auto;
}

.status-buttons > .btn {
  padding: 0.4rem 0.7rem;
  font-size: 1rem;
}

/* Step/echo control rows: every button (toggles, play, stop) shares one
   compact size so the two rows look identical. */
.echo-controls .btn {
  font-size: 1rem;
  padding: 0.3rem 0.8rem;
}

.btn.ar,
.btn.diff,
.btn.status.play {

}

/* The play button reuses .btn.status (spinner styling) but, unlike a plain
   status readout, it is clickable. */
.btn.status.play:not(:disabled):hover {
  background-color: var(--color-accent);
  cursor: pointer;
}

.btn.status .spinner {
  display: inline-block;
  width: 0.75em;
  height: 0.75em;
  border: 2px solid currentColor;
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.btn.status:hover {
  background-color: var(--color-secondary);
  cursor: auto;
}

.btn.status.done {
  color: var(--color-accent);
  border-bottom-color: var(--color-accent);
}
.btn.status.warning {
  color: var(--color-waring);
  border-bottom-color: var(--color-warning);
}

.btn.stop {
  color: var(--color-error);
  border-bottom-color: var(--color-error);
  opacity: 0.5;
}

.btn.stop:not(:disabled) {
  opacity: 1;
}

.btn.stop:not(:disabled):hover {
  background-color: var(--color-error);
  color: var(--color-secondary);
}

.toggle {
  padding: 0.3rem 0.8rem;
}

.toggle-active {
  background-color: var(--color-accent);
  border-bottom-color: var(--color-primary);
}

.tok {
  color: var(--color-primary);
  background-color: var(--color-accent);
}

.tok-selected {
  outline: 3px solid var(--color-primary);
  outline-offset: -1px;
}

/* Attention timescale: prompt context tokens — present and attended-to, but
   muted relative to the response so the generated text still reads clearly. */
.tok-prompt, .tok-special {
  background-color: var(--color-secondary);
}

/* Special / chat-template tokens (e.g. <|start_header_id|>) — shown because the
   model attends over them, but rendered small + dimmed as clear artefacts. */
.tok-special {
}

/* login overlay */
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
  color: var(--color-error);
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
  margin-top: 5rem;
  max-width: 250px;
}
</style>
