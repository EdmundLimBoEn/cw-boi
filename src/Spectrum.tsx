import { useEffect, useRef } from 'react'
import type { DecoderReading } from './decoder'

export function Spectrum({ reading, active, frequency, bandwidth, onTune }: { reading: DecoderReading; active: boolean; frequency: number; bandwidth: number; onTune: (frequency: number) => void }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const history = useRef<number[][]>([])
  useEffect(() => {
    if (active && reading.spectrum.length) {
      history.current.unshift(reading.spectrum)
      history.current.length = Math.min(history.current.length, 64)
    }
    const element = canvas.current
    if (!element) return
    const context = element.getContext('2d')
    if (!context) return
    const draw = () => {
      const { width, height } = element.getBoundingClientRect()
      const dpr = window.devicePixelRatio || 1
      element.width = width * dpr; element.height = height * dpr
      context.scale(dpr, dpr)
      context.fillStyle = '#172b39'; context.fillRect(0, 0, width, height)
      const left = 33, right = width - 16, top = 20, bottom = height * 0.56
      const x = (hz: number) => left + (hz - 200) / 1300 * (right - left)
      context.lineWidth = 1
      for (let db = -80; db <= -20; db += 20) {
        const y = top + (-db - 10) / 80 * (bottom - top)
        context.strokeStyle = '#29404e'; context.beginPath(); context.moveTo(left, y); context.lineTo(right, y); context.stroke()
        context.fillStyle = '#68818e'; context.font = '9px "IBM Plex Mono", monospace'; context.fillText(String(db), 5, y + 3)
      }
      for (let hz = 300; hz <= 1400; hz += 200) {
        context.strokeStyle = '#29404e'; context.beginPath(); context.moveTo(x(hz), top); context.lineTo(x(hz), height - 23); context.stroke()
        context.fillStyle = '#8ba0aa'; context.fillText(String(hz), x(hz) - 10, height - 8)
      }
      const tuningX = x(frequency), tuningWidth = bandwidth / 1300 * (right - left)
      context.fillStyle = '#75d7e911'; context.fillRect(tuningX - tuningWidth / 2, top, tuningWidth, height - 43)
      const spectrum = active ? reading.spectrum : []
      context.beginPath()
      if (spectrum.length) {
        spectrum.forEach((db, i) => {
          const px = left + i / (spectrum.length - 1) * (right - left)
          const py = top + Math.min(80, Math.max(0, -db - 10)) / 80 * (bottom - top)
          if (i === 0) context.moveTo(px, py); else context.lineTo(px, py)
        })
      } else { context.moveTo(left, bottom - 1); context.lineTo(right, bottom - 1) }
      context.strokeStyle = '#87e4ef'; context.lineWidth = 1.5; context.stroke()
      const waterfallTop = bottom + 12, waterfallBottom = height - 27
      context.fillStyle = '#10232f'; context.fillRect(left, waterfallTop, right - left, waterfallBottom - waterfallTop)
      history.current.forEach((row, rowIndex) => {
        row.forEach((db, i) => {
          const strength = Math.min(1, Math.max(0, (db + 85) / 70))
          if (strength < 0.06) return
          context.fillStyle = `rgba(${Math.round(32 + strength * 112)}, ${Math.round(70 + strength * 163)}, ${Math.round(104 + strength * 132)}, ${0.22 + strength * 0.78})`
          context.fillRect(left + i / row.length * (right - left), waterfallTop + rowIndex / 64 * (waterfallBottom - waterfallTop), (right - left) / row.length + 0.5, (waterfallBottom - waterfallTop) / 64 + 0.5)
        })
      })
      context.strokeStyle = '#e4b78a'; context.lineWidth = 1; context.setLineDash([3, 4])
      context.beginPath(); context.moveTo(tuningX, top); context.lineTo(tuningX, height - 24); context.stroke(); context.setLineDash([])
      context.fillStyle = '#e4b78a'; context.beginPath(); context.moveTo(tuningX - 4, top - 6); context.lineTo(tuningX + 4, top - 6); context.lineTo(tuningX, top); context.fill()
    }
    draw()
    const observer = new ResizeObserver(draw)
    observer.observe(element)
    return () => observer.disconnect()
  }, [reading, active, frequency, bandwidth])
  return <canvas ref={canvas} className="spectrum-canvas" role="slider" aria-label="Receiver tuning frequency" aria-valuemin={250} aria-valuemax={1400} aria-valuenow={Math.round(frequency)} aria-valuetext={`${Math.round(frequency)} hertz`} tabIndex={0}
    onClick={event => { const rect = event.currentTarget.getBoundingClientRect(); onTune(Math.round(Math.min(1400, Math.max(250, 200 + (event.clientX - rect.left - 33) / (rect.width - 49) * 1300)) / 5) * 5) }}
    onKeyDown={event => { if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') { event.preventDefault(); onTune(Math.min(1400, Math.max(250, frequency + (event.key === 'ArrowRight' ? 5 : -5)))) } }} />
}
