<template>
  <div class="heartbeat-wrap">
    <!-- model badge (top-left) -->
    <div v-if="modelLabel" class="model-badge">{{ modelLabel }}</div>

    <!-- network toggle (top-right) -->
    <button
      class="net-toggle"
      :class="{ active: showNetwork }"
      @click="showNetwork = !showNetwork"
      title="Show network request rhythm"
    >net</button>

    <svg
      class="heartbeat"
      viewBox="0 0 1000 120"
      preserveAspectRatio="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <!-- inference line -->
      <path v-if="path" :d="path" fill="none" stroke="#832161" stroke-width="4" vector-effect="non-scaling-stroke" />
      <!-- network requests line -->
      <path
        v-if="showNetwork && netPath"
        :d="netPath"
        fill="none"
        stroke="#832161"
        stroke-width="2"
        stroke-dasharray="6 4"
        opacity="0.4"
        vector-effect="non-scaling-stroke"
      />
    </svg>

    <!-- step-interval tick marks -->
    <svg
      v-if="tickPositions.length"
      class="tick-row"
      viewBox="0 0 1000 18"
      preserveAspectRatio="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <line
        v-for="(tx, i) in tickPositions"
        :key="i"
        :x1="tx" y1="2" :x2="tx" y2="14"
        stroke="#832161"
        stroke-width="4"
        opacity="1"
        vector-effect="non-scaling-stroke"
      />
    </svg>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

const props = defineProps<{
  points: { x: number; y: number }[]
  netPoints?: { x: number; y: number }[]
  stepTimes?: number[]   // wall-clock timestamps (seconds) of each inference step
  modelLabel?: string    // e.g. "llama-3.2-1B · AR · 12 tok/s"
}>()

const W = 1000
const H = 120
const PAD = 3

const showNetwork = ref(false)

function buildPath(pts: { x: number; y: number }[]): string {
  if (pts.length < 2) return ''
  const xMin = pts[0].x
  const xMax = pts[pts.length - 1].x
  const xRange = xMax - xMin || 1
  const yMin = Math.min(...pts.map(p => p.y))
  const yMax = Math.max(...pts.map(p => p.y))
  const yRange = yMax - yMin || 1
  return pts
    .map((p, i) => {
      const sx = ((p.x - xMin) / xRange) * W
      const sy = H - PAD - ((p.y - yMin) / yRange) * (H - PAD * 2)
      return `${i === 0 ? 'M' : 'L'}${sx.toFixed(2)},${sy.toFixed(2)}`
    })
    .join(' ')
}

const path = computed(() => buildPath(props.points))
const netPath = computed(() => buildPath(props.netPoints ?? []))

// Map step timestamps onto the same x-axis as the inference line
const tickPositions = computed(() => {
  const times = props.stepTimes
  const pts = props.points
  if (!times || times.length < 1 || pts.length < 2) return []
  const xMin = pts[0].x
  const xMax = pts[pts.length - 1].x
  const xRange = xMax - xMin || 1
  return times.map(t => ((t - xMin) / xRange) * W)
})
</script>

<style scoped>
.heartbeat-wrap {
  position: relative;
}

.heartbeat {
  width: 100%;
  height: 120px;
  background: transparent;
  display: block;
}

.tick-row {
  width: 100%;
  height: 18px;
  display: block;
  margin-top: -2px;
}

/* model badge — top-left */
.model-badge {
  position: absolute;
  top: 7px;
  left: 10px;
  z-index: 2;
  font-family: var(--font-sans, sans-serif);
  font-size: 0.62rem;
  letter-spacing: 0.04em;
  color: #832161;
  opacity: 0.55;
  pointer-events: none;
  white-space: nowrap;
}

/* network toggle — top-right */
.net-toggle {
  position: absolute;
  top: 6px;
  right: 8px;
  z-index: 2;
  font-family: var(--font-sans, sans-serif);
  font-size: 0.62rem;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  padding: 2px 7px;
  background: transparent;
  color: #832161;
  border: 1.5px dashed #832161;
  border-bottom: 1.5px dashed #832161;
  border-radius: 3px;
  opacity: 0.4;
  cursor: pointer;
  margin: 0;
  font-weight: 400;
  transition: opacity 0.15s, background 0.15s;
}

.net-toggle:hover,
.net-toggle.active {
  opacity: 1;
  background: rgba(131, 33, 97, 0.07);
}
</style>
