<template>
  <svg
    class="heartbeat"
    viewBox="0 0 1000 60"
    preserveAspectRatio="none"
    xmlns="http://www.w3.org/2000/svg"
  >
    <path v-if="path" :d="path" fill="none" stroke="#fff" stroke-width="1.5" vector-effect="non-scaling-stroke" />
  </svg>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  points: { x: number; y: number }[]
}>()

const W = 1000
const H = 60
const PAD = 3

const path = computed(() => {
  const pts = props.points
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
})
</script>

<style scoped>
.heartbeat {
  width: 100%;
  height: 60px;
  background: #000;
  border-radius: 6px;
  display: block;
}
</style>

