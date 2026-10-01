import { decodeTiming } from './timing'

export type ReceiverSettings = { frequency: number; autoTune: boolean; bandwidth: number; threshold: number; wpm: number; autoSpeed: boolean; spacing: number }
export const DEFAULT_RECEIVER: ReceiverSettings = { frequency: 650, autoTune: true, bandwidth: 100, threshold: 6, wpm: 20, autoSpeed: true, spacing: 20 }
export type DecoderReading = { text: string; pending: string; frequency: number; wpm: number; snr: number; level: number; keyed: boolean; confidence: number; spectrum: number[]; alternatives?: string[] }
export type Run = { on: boolean; duration: number }
const clamp = (n: number, low: number, high: number) => Math.min(high, Math.max(low, n))
const median = (xs: number[]) => xs.length ? [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)] : 0

export function inferDit(runs: Run[], initial: number) {
  const marks = runs.filter(run => run.on && run.duration > 0.012 && run.duration < 0.9).map(run => run.duration).slice(-120)
  const gaps = runs.filter(run => !run.on && run.duration > 0.01).map(run => run.duration).slice(-120)
  if (marks.length < 2) return initial
  const sortedGaps = [...gaps].sort((a, b) => a - b)
  const shortestGap = sortedGaps[Math.floor(sortedGaps.length * 0.2)]
  let best = initial, bestCost = Infinity
  for (let dit = 0.02; dit <= 0.245; dit += 0.001) {
    let cost = 0
    for (const duration of marks) {
      const error = Math.min(Math.abs(Math.log(duration / dit)), Math.abs(Math.log(duration / (3 * dit))))
      cost += Math.min(error * error, 1)
    }
    cost /= marks.length
    if (shortestGap) cost += 0.28 * Math.min(Math.abs(Math.log(shortestGap / dit)) ** 2, 1)
    cost += 0.003 * Math.abs(Math.log(dit / initial))
    if (cost < bestCost) { bestCost = cost; best = dit }
  }
  return best
}

// Radix-2 FFT shared by live reception, file decoding, and the reproducible benchmark.
export function spectrumPower(samples: Float32Array): Float32Array {
  const n = samples.length, real = new Float32Array(n), imag = new Float32Array(n)
  for (let i = 0, j = 0; i < n; i++) {
    real[j] = samples[i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1)))
    let bit = n >> 1
    while (j & bit) { j ^= bit; bit >>= 1 }
    j ^= bit
  }
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1, angle = -2 * Math.PI / size
    for (let offset = 0; offset < n; offset += size) {
      for (let j = 0; j < half; j++) {
        const c = Math.cos(angle * j), s = Math.sin(angle * j), k = offset + j, l = k + half
        const tr = c * real[l] - s * imag[l], ti = s * real[l] + c * imag[l]
        real[l] = real[k] - tr; imag[l] = imag[k] - ti
        real[k] += tr; imag[k] += ti
      }
    }
  }
  const power = new Float32Array(n / 2)
  for (let i = 0; i < power.length; i++) power[i] = (real[i] ** 2 + imag[i] ** 2) * 16 / n ** 2
  return power
}

export class CWDecoder {
  settings: ReceiverSettings
  frequency: number
  private sampleRate: number
  private frame = new Float32Array(1024)
  private frameIndex = 0
  private frameCount = 0
  private sampleCount = 0
  private hopCount = 0
  private phase = 0
  private i1 = 0
  private q1 = 0
  private i2 = 0
  private q2 = 0
  private noise = 0.000001
  private peak = 0.00001
  private keyed = false
  private candidate = false
  private candidateSince = 0
  private lastEdge = 0
  private lastDecode = 0
  private lastMark = 0
  private runs: Run[] = []
  private dit: number
  private lastReading: DecoderReading
  private nextScan = 0
  private scanCandidate = 0
  private scanCount = 0
  private spectrum: number[] = []
  private lastSnr = 0
  private markI = 0
  private markQ = 0
  private markSamples = 0
  private carrierUntil = 0
  private power = 0
  private bootstrap: number[] = []
  private primed = false
  private committed = ''

  constructor(sampleRate: number, settings: Partial<ReceiverSettings> = {}) {
    this.sampleRate = sampleRate
    this.settings = { ...DEFAULT_RECEIVER, ...settings }
    this.frequency = this.settings.frequency
    this.dit = 1.2 / this.settings.wpm
    this.lastReading = { text: '', pending: '', frequency: this.frequency, wpm: this.settings.wpm, snr: 0, level: -100, keyed: false, confidence: 0, spectrum: [] }
  }

  configure(settings: Partial<ReceiverSettings>) {
    Object.assign(this.settings, settings)
    if (settings.frequency !== undefined && !this.settings.autoTune) this.frequency = settings.frequency
    if (settings.wpm !== undefined && !this.settings.autoSpeed) this.dit = 1.2 / settings.wpm
  }

  process(samples: Float32Array) {
    if (!this.primed) {
      for (const sample of samples) this.bootstrap.push(sample)
      if (this.bootstrap.length < this.sampleRate * 0.256) return this.lastReading
      samples = Float32Array.from(this.bootstrap)
      this.prime(samples)
      this.primed = true
      this.bootstrap = []
    }
    const alpha = 1 - Math.exp(-2 * Math.PI * (this.settings.bandwidth / 2) / this.sampleRate)
    const hop = Math.max(1, Math.round(this.sampleRate * 0.0025))
    for (let index = 0; index < samples.length; index++) {
      const sample = Number.isFinite(samples[index]) ? samples[index] : 0
      this.frame[this.frameIndex] = sample
      this.frameIndex = (this.frameIndex + 1) % this.frame.length
      this.frameCount++
      this.phase += 2 * Math.PI * this.frequency / this.sampleRate
      if (this.phase > 2 * Math.PI) this.phase -= 2 * Math.PI
      this.i1 += alpha * (sample * Math.cos(this.phase) - this.i1)
      this.q1 += alpha * (sample * Math.sin(this.phase) - this.q1)
      this.i2 += alpha * (this.i1 - this.i2)
      this.q2 += alpha * (this.q1 - this.q2)
      if (this.keyed) {
        this.markI += sample * Math.cos(this.phase)
        this.markQ += sample * Math.sin(this.phase)
        this.markSamples++
      }
      this.sampleCount++
      if (++this.hopCount >= hop) {
        this.hopCount = 0
        const time = this.sampleCount / this.sampleRate
        if (time >= this.nextScan && this.frameCount >= this.frame.length) {
          this.scan(time)
          this.nextScan = time + 0.04
        }
        this.detect(4 * (this.i2 ** 2 + this.q2 ** 2), time)
      }
    }
    const time = this.sampleCount / this.sampleRate
    if (time - this.lastDecode >= 0.05) {
      this.lastDecode = time
      this.updateText(time)
    }
    return this.lastReading
  }

  private detect(power: number, time: number) {
    this.power = power
    this.peak = Math.max(power, this.peak * 0.982)
    // A Schmitt gate follows the signal envelope; the noise estimate only learns below the gate.
    const floor = Math.max(1e-9, this.noise * 10 ** (this.settings.threshold / 10))
    const threshold = Math.max(floor, this.peak * (this.keyed ? 0.14 : 0.27))
    const on = power > threshold
    if (!on && power < this.peak * 0.1) this.noise += 0.003 * (power - this.noise)
    if (on !== this.candidate) { this.candidate = on; this.candidateSince = time }
    if (on !== this.keyed && time - this.candidateSince >= Math.min(0.009, this.dit * 0.18)) {
      const edge = this.candidateSince
      const alpha = 1 - Math.exp(-Math.PI * this.settings.bandwidth / this.sampleRate)
      const coherence = (this.markI ** 2 + this.markQ ** 2) * alpha / Math.max(1e-12, this.noise * this.markSamples)
      if (this.keyed && coherence >= 10 ** ((this.settings.threshold + 4) / 10)) this.carrierUntil = time + 1.2
      if (this.keyed && time > this.carrierUntil) {
        // Carrier hysteresis preserves faded elements while rejecting noise-only excursions.
        const gap = this.runs.at(-1)
        if (gap && !gap.on) { this.lastEdge -= gap.duration; this.runs.pop() }
        this.keyed = false
        return
      }
      if (this.lastEdge || this.keyed) this.runs.push({ on: this.keyed, duration: edge - this.lastEdge })
      if (this.keyed) this.lastMark = edge
      this.keyed = on
      if (on) { this.markI = 0; this.markQ = 0; this.markSamples = 0 }
      this.lastEdge = edge
      if (!on && this.settings.autoSpeed) this.dit = inferDit(this.runs, 1.2 / this.settings.wpm)
    }
  }

  private scan(time: number) {
    const frame = new Float32Array(this.frame.length)
    for (let i = 0; i < frame.length; i++) frame[i] = this.frame[(this.frameIndex + i) % frame.length]
    const power = spectrumPower(frame)
    const hz = this.sampleRate / frame.length
    const low = Math.ceil(250 / hz), high = Math.min(power.length - 2, Math.floor(1400 / hz))
    const band = Array.from(power.slice(low, high + 1))
    const background = Math.max(1e-12, median(band))
    const alpha = 1 - Math.exp(-Math.PI * this.settings.bandwidth / this.sampleRate)
    const estimate = background * this.frame.length / (6 * Math.LN2) * alpha
    this.noise = Math.max(1e-10, this.noise * 0.85 + estimate * 0.15)
    let peakBin = low
    for (let i = low + 1; i <= high; i++) if (power[i] > power[peakBin]) peakBin = i
    const prominence = 10 * Math.log10(Math.max(1e-12, power[peakBin]) / background)
    this.spectrum = Array.from({ length: 160 }, (_, i) => {
      const bin = Math.round((200 + i * 1300 / 159) / hz)
      return 10 * Math.log10(Math.max(1e-12, power[bin] ?? 0))
    })
    const targetBin = Math.round(this.frequency / hz)
    this.lastSnr = clamp(10 * Math.log10(Math.max(1e-12, power[targetBin]) / background), 0, 60)
    if (!this.settings.autoTune || prominence < this.settings.threshold + 7 || power[peakBin] < 1e-8) return
    const l = Math.log(Math.max(power[peakBin - 1], 1e-15)), m = Math.log(Math.max(power[peakBin], 1e-15)), r = Math.log(Math.max(power[peakBin + 1], 1e-15))
    const delta = clamp(0.5 * (l - r) / (l - 2 * m + r || 1), -0.5, 0.5)
    const candidate = (peakBin + delta) * hz
    if (Math.abs(candidate - this.scanCandidate) < 35) this.scanCount++
    else { this.scanCandidate = candidate; this.scanCount = 1 }
    const close = Math.abs(candidate - this.frequency) < this.settings.bandwidth / 2
    if (close || (this.scanCount >= 2 && (!this.keyed || time - this.lastMark > 1))) {
      this.frequency = close ? this.frequency * 0.65 + candidate * 0.35 : candidate
    }
  }

  private prime(samples: Float32Array) {
    let strongest = 0, carrier = this.frequency
    const floors: number[] = []
    const n = this.frame.length, hz = this.sampleRate / n
    for (let offset = 0; offset + n <= samples.length; offset += n / 2) {
      const power = spectrumPower(samples.slice(offset, offset + n))
      const low = Math.ceil(250 / hz), high = Math.floor(1400 / hz)
      const background = Math.max(1e-12, median(Array.from(power.slice(low, high))))
      floors.push(background)
      let peakBin = low
      for (let i = low + 1; i <= high; i++) if (power[i] > power[peakBin]) peakBin = i
      if (power[peakBin] > strongest && power[peakBin] / background > 10 ** ((this.settings.threshold + 7) / 10)) {
        strongest = power[peakBin]
        const l = Math.log(Math.max(power[peakBin - 1], 1e-15)), m = Math.log(Math.max(power[peakBin], 1e-15)), r = Math.log(Math.max(power[peakBin + 1], 1e-15))
        carrier = (peakBin + clamp(0.5 * (l - r) / (l - 2 * m + r || 1), -0.5, 0.5)) * hz
      }
    }
    if (this.settings.autoTune && strongest > 1e-8) this.frequency = carrier
    this.noise = Math.max(1e-10, median(floors) * n / (6 * Math.LN2) * (1 - Math.exp(-Math.PI * this.settings.bandwidth / this.sampleRate)))
  }

  private updateText(time: number, final = false) {
    const speed = this.settings.autoSpeed ? 1.2 / this.dit : this.settings.wpm
    const spacingSpeed = Math.min(speed, this.settings.spacing)
    const gapUnit = this.settings.autoSpeed && this.settings.spacing === this.settings.wpm ? this.dit : (60 / spacingSpeed - 31 * this.dit) / 19
    if (this.runs.length > 900) {
      let boundary = this.runs.findIndex((run, index) => index > 500 && !run.on && run.duration > gapUnit * 5)
      if (boundary < 0) boundary = this.runs.findIndex((run, index) => index > 500 && !run.on && run.duration > gapUnit * 1.8)
      // An endless unspaced noise burst must not grow the live beam search without bound.
      if (boundary < 0) { this.runs.splice(0, 600); this.committed += '�' }
      else this.committed += decodeTiming(this.runs.splice(0, boundary + 1), this.dit, gapUnit, this.settings.autoSpeed, true).text
    }
    const runs = this.runs.slice()
    if (this.lastEdge) runs.push({ on: this.keyed, duration: time - this.lastEdge })
    const decoded = decodeTiming(runs, this.dit, gapUnit, this.settings.autoSpeed, final)
    this.lastReading = { ...decoded, text: this.committed + decoded.text, frequency: Math.round(this.frequency), wpm: Math.round(1.2 / this.dit), snr: Math.round(this.lastSnr), level: Math.round(10 * Math.log10(Math.max(this.power, 1e-10))), keyed: this.keyed, spectrum: this.spectrum }
  }

  finish() {
    if (!this.primed && this.bootstrap.length) {
      this.primed = true
      const samples = Float32Array.from(this.bootstrap)
      this.bootstrap = []
      this.prime(samples)
      this.process(samples)
    }
    this.updateText(this.sampleCount / this.sampleRate, true)
    return this.lastReading
  }
}

export function decodeSamples(samples: Float32Array, sampleRate: number, settings: Partial<ReceiverSettings> = {}) {
  const decoder = new CWDecoder(sampleRate, settings)
  for (let i = 0; i < samples.length; i += 256) decoder.process(samples.subarray(i, i + 256))
  return decoder.finish()
}
