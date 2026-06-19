<template>
  <div class="heartbeat-wrap">
    <!-- model badge (top-left) -->
    <div v-if="modelLabel" class="model-badge">{{ modelLabel }}</div>
    <!-- Y-axis labels -->
    <div class="axis-labels">
      <div class="top-left" v-if="!isAttentionTimescale">{{ yAxisMax }} tokens</div>
      <div class="top-left" v-else>DLA</div>
      <!-- X-axis labels -->
      <div class="bottom-left">0ms</div>
      <div class="bottom-right">{{ xAxisMaxMs }}ms</div>
    </div>

    <svg
      class="heartbeat"
      viewBox="0 0 1000 140"
      preserveAspectRatio="none"
      xmlns="http://www.w3.org/2000/svg"
    >

      <!-- zero / baseline reference line -->
      <line
        v-if="baselineY !== null"
        x1="0" :y1="baselineY" x2="1000" :y2="baselineY"
        stroke="#832161"
        stroke-width="1.5"
        stroke-dasharray="4 5"
        opacity="0.45"
        vector-effect="non-scaling-stroke"
      />
      <!-- inference line -->
      <path v-if="path" :d="path" fill="none" stroke="#832161" stroke-width="4" vector-effect="non-scaling-stroke" />
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

    <!-- active-layer cursor (attention playback): subtle dot sweeping L→R -->
    <div
      v-if="cursorLeftPct !== null"
      class="layer-cursor"
      :style="{ left: cursorLeftPct + '%' }"
    ></div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

const props = defineProps<{
  points: { x: number; y: number }[]
  stepTimes?: number[]   // wall-clock timestamps (seconds) of each inference step
  modelLabel?: string    // e.g. "llama-3.2-1B · AR · 12 tok/s"
  activeIndex?: number   // index of the point currently highlighted (-1/undefined = none)
  baseline?: number | null  // draw a horizontal reference line at this y value (e.g. 0)
  isAttentionTimescale?: false
}>()

const W = 1000
const H = 140
const PAD = 3

// Scale a point series to viewBox coords; also return each point's x so the
// active-layer cursor can sit exactly on its data point, plus the mapped y of an
// optional baseline value (which is folded into the y-range so it stays visible).
function scalePoints(
  pts: { x: number; y: number }[],
  baseline?: number | null,
): { d: string; xs: number[]; baselineY: number | null } {
  if (pts.length < 1) return { d: '', xs: [], baselineY: null }
  const xMin = pts[0].x
  const xMax = pts[pts.length - 1].x
  const xRange = xMax - xMin || 1
  let yMin = Math.min(...pts.map(p => p.y))
  let yMax = Math.max(...pts.map(p => p.y))
  const hasBase = baseline !== undefined && baseline !== null
  if (hasBase) { yMin = Math.min(yMin, baseline!); yMax = Math.max(yMax, baseline!) }
  const yRange = yMax - yMin || 1
  const mapY = (y: number) => H - PAD - ((y - yMin) / yRange) * (H - PAD * 2)
  const xs: number[] = []
  const segs = pts.map((p, i) => {
    const sx = ((p.x - xMin) / xRange) * W
    xs.push(sx)
    return `${i === 0 ? 'M' : 'L'}${sx.toFixed(2)},${mapY(p.y).toFixed(2)}`
  })
  return {
    d: pts.length < 2 ? '' : segs.join(' '),
    xs,
    baselineY: hasBase ? mapY(baseline!) : null,
  }
}

const scaled = computed(() => scalePoints(props.points, props.baseline))
const path = computed(() => scaled.value.d)
const baselineY = computed(() => scaled.value.baselineY)

// Horizontal position (% of width) of the currently-sonified layer's dot.
const cursorLeftPct = computed<number | null>(() => {
  const i = props.activeIndex
  if (i === undefined || i < 0) return null
  const xs = scaled.value.xs
  if (!xs.length) return null
  return (xs[Math.min(i, xs.length - 1)] / W) * 100
})

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

const xAxisMaxMs = computed(() => {
  const times = props.stepTimes
  if (!times?.length) return 0

  return Math.round((times[times.length - 1] - times[0]) * 1000)
})

const yAxisMax = computed(() => props.points.length)
</script>

<style scoped>
.heartbeat-wrap {
  position: relative;
}

.heartbeat {
  width: 100%;
  height: 140px;
  background: transparent;
  display: block;
}

.tick-row {
  width: 100%;
  height: 18px;
  display: block;
  margin-top: -2px;
}

/* active-layer cursor — subtle dot along the bottom, animates L→R */
.layer-cursor {
  position: absolute;
  bottom: 1px;
  width: 9px;
  height: 9px;
  margin-left: -4.5px;     /* centre on the data point */
  border-radius: 50%;
  background: #832161;
  opacity: 0.55;
  pointer-events: none;
  transition: left 0.3s ease;
}

/* model badge — top-left */
.model-badge {
  z-index: 2;
  font-family: var(--font-sans, sans-serif);
  font-size: 1.2rem;
  color: var(--color-primary);
  pointer-events: none;
  white-space: nowrap;
}

.axis-labels {
  opacity: .5;
  color: var(--color-primary);
  position: absolute;
  width: 100%;
  height: 100%;
}

.axis-labels .bottom-left {
  position: absolute;
  bottom: 8px;
}

.axis-labels .bottom-right {
  position: absolute;
  bottom: 8px;
  right: 0;
}
</style>
