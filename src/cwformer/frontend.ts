// Matches neural/cwformer_engine.py: complex carrier filter, 2x FIR, then block AGC.
const TAU = 2 * Math.PI

function besselI0(x: number) {
  let sum = 1, term = 1
  for (let k = 1; k < 40; k++) {
    term *= x * x / (4 * k * k)
    sum += term
    if (term < sum * 1e-17) break
  }
  return sum
}

function resamplerTaps() {
  const taps = new Float64Array(41), scale = besselI0(5)
  let total = 0
  for (let i = 0; i < taps.length; i++) {
    const offset = i - 20
    const sinc = offset === 0 ? 0.5 : Math.sin(Math.PI * 0.5 * offset) / (Math.PI * offset)
    taps[i] = sinc * besselI0(5 * Math.sqrt(1 - (offset / 20) ** 2)) / scale
    total += taps[i]
  }
  for (let i = 0; i < taps.length; i++) taps[i] *= 2 / total
  return taps
}

export class Frontend {
  private readonly sections: number[][]
  private readonly filterReal = new Float64Array(4)
  private readonly filterImag = new Float64Array(4)
  private readonly taps = resamplerTaps()
  private readonly resampleState = new Float64Array(40)
  private phase = 0
  private outputPhase = 0
  private peak = 0

  constructor(public frequency: number, public readonly bandwidth: number) {
    if (!Number.isFinite(frequency) || !Number.isFinite(bandwidth) || bandwidth <= 0 || bandwidth >= 8000) {
      throw new Error('Invalid carrier filter settings')
    }
    const k = Math.tan(Math.PI * bandwidth / 16000), denominator = 1 + k + k * k
    const gain = k ** 3 / ((1 + k) * denominator)
    // scipy's odd-order SOS pairs two zeros with the real pole in the first section.
    this.sections = [
      [gain, 2 * gain, gain, (k - 1) / (k + 1), 0],
      [1, 1, 0, 2 * (k * k - 1) / denominator, (1 - k + k * k) / denominator],
    ]
  }

  process(audio: Float32Array): Float32Array {
    if (!Number.isFinite(this.frequency)) throw new Error('Invalid carrier frequency')
    const result = new Float64Array(audio.length * 2)
    const step = TAU * this.frequency / 8000, outputStep = TAU * 650 / 8000
    let blockPeak = 0
    for (let i = 0; i < audio.length; i++) {
      const phase = this.phase + i * step
      let real = audio[i] * Math.cos(phase) * 2, imag = -audio[i] * Math.sin(phase) * 2
      for (let section = 0; section < 2; section++) {
        const [b0, b1, b2, a1, a2] = this.sections[section], at = section * 2
        const nextReal = b0 * real + this.filterReal[at], nextImag = b0 * imag + this.filterImag[at]
        this.filterReal[at] = b1 * real - a1 * nextReal + this.filterReal[at + 1]
        this.filterImag[at] = b1 * imag - a1 * nextImag + this.filterImag[at + 1]
        this.filterReal[at + 1] = b2 * real - a2 * nextReal
        this.filterImag[at + 1] = b2 * imag - a2 * nextImag
        real = nextReal
        imag = nextImag
      }
      const outputPhase = this.outputPhase + i * outputStep
      const shifted = real * Math.cos(outputPhase) - imag * Math.sin(outputPhase)
      for (let upsample = 0; upsample < 2; upsample++) {
        const sample = upsample === 0 ? shifted : 0
        const value = this.taps[0] * sample + this.resampleState[0]
        for (let j = 0; j < 39; j++) this.resampleState[j] = this.taps[j + 1] * sample + this.resampleState[j + 1]
        this.resampleState[39] = this.taps[40] * sample
        result[i * 2 + upsample] = value
        blockPeak = Math.max(blockPeak, Math.abs(value))
      }
    }
    this.phase = ((this.phase + audio.length * step) % TAU + TAU) % TAU
    this.outputPhase = (this.outputPhase + audio.length * outputStep) % TAU
    this.peak = Math.max(blockPeak, this.peak * Math.exp(-audio.length / 16000))
    const gain = 0.7 / Math.max(this.peak, 1e-5)
    return Float32Array.from(result, value => value * gain)
  }
}

const FFT_SIZE = 400, FFT_BINS = 201, MEL_BINS = 40, HOP = 160

// A mixed-radix FFT keeps the model's 400-point transform without zero-padding it.
class Fourier400 {
  private readonly cosine = Float64Array.from({ length: FFT_SIZE }, (_, i) => Math.cos(-TAU * i / FFT_SIZE))
  private readonly sine = Float64Array.from({ length: FFT_SIZE }, (_, i) => Math.sin(-TAU * i / FFT_SIZE))
  private readonly real = new Float64Array(FFT_SIZE)
  private readonly imag = new Float64Array(FFT_SIZE)
  private readonly scratchReal = new Float64Array(FFT_SIZE)
  private readonly scratchImag = new Float64Array(FFT_SIZE)
  readonly power = new Float64Array(FFT_BINS)

  private transform(input: Float32Array, start: number, stride: number, size: number, at: number) {
    if (size === 1) {
      this.real[at] = input[start]
      this.imag[at] = 0
      return
    }
    const radix = size % 2 === 0 ? 2 : 5, part = size / radix
    for (let j = 0; j < radix; j++) this.transform(input, start + j * stride, stride * radix, part, at + j * part)
    for (let k = 0; k < size; k++) {
      let real = 0, imag = 0
      for (let j = 0; j < radix; j++) {
        const from = at + j * part + k % part, twiddle = j * k * (FFT_SIZE / size) % FFT_SIZE
        const c = this.cosine[twiddle], s = this.sine[twiddle]
        real += this.real[from] * c - this.imag[from] * s
        imag += this.real[from] * s + this.imag[from] * c
      }
      this.scratchReal[at + k] = real
      this.scratchImag[at + k] = imag
    }
    for (let i = at; i < at + size; i++) {
      this.real[i] = this.scratchReal[i]
      this.imag[i] = this.scratchImag[i]
    }
  }

  process(input: Float32Array) {
    this.transform(input, 0, 1, FFT_SIZE, 0)
    for (let i = 0; i < FFT_BINS; i++) this.power[i] = this.real[i] ** 2 + this.imag[i] ** 2
    return this.power
  }
}

export class MelFrontend {
  private overlap: Float32Array = new Float32Array(200)
  private readonly frame = new Float32Array(FFT_SIZE)
  private readonly fourier = new Fourier400()

  constructor(private readonly window: Float32Array, private readonly basis: Float32Array) {
    if (window.length !== FFT_SIZE || basis.length !== MEL_BINS * FFT_BINS) throw new Error('Invalid CWformer mel assets')
  }

  process(audio: Float32Array): { data: Float32Array; frames: number } {
    const joined = new Float32Array(this.overlap.length + audio.length)
    joined.set(this.overlap)
    joined.set(audio, this.overlap.length)
    const frames = Math.max(0, Math.floor((joined.length - FFT_SIZE) / HOP) + 1)
    this.overlap = joined.slice(frames * HOP)
    const data = new Float32Array(frames * MEL_BINS)
    for (let frame = 0; frame < frames; frame++) {
      for (let i = 0; i < FFT_SIZE; i++) this.frame[i] = joined[frame * HOP + i] * this.window[i]
      const power = this.fourier.process(this.frame)
      for (let mel = 0; mel < MEL_BINS; mel++) {
        let energy = 0
        for (let bin = 0; bin < FFT_BINS; bin++) energy += power[bin] * this.basis[mel * FFT_BINS + bin]
        data[frame * MEL_BINS + mel] = Math.log(energy + 1e-6)
      }
    }
    return { data, frames }
  }
}
