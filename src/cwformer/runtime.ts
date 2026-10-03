import * as ort from 'onnxruntime-web/wasm'
import { Frontend, MelFrontend } from './frontend'

ort.env.wasm.numThreads = 1
ort.env.wasm.proxy = false

export const TOKENS = ['', ' ', ...'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?/(&=+', 'AR', 'SK', 'BT', 'KN', 'AS', 'CT']
export type Settings = { frequency: number; bandwidth: number }

export function validateSettings(settings: Settings) {
  if (!Number.isFinite(settings.frequency) || settings.frequency < 250 || settings.frequency > 1400) throw new Error('Tone must be between 250 and 1400 Hz.')
  if (!Number.isFinite(settings.bandwidth) || settings.bandwidth < 40 || settings.bandwidth > 500) throw new Error('Filter width must be between 40 and 500 Hz.')
}

const zeros = (dims: number[]) => new ort.Tensor('float32', new Float32Array(dims.reduce((a, b) => a * b, 1)), dims)
const release = (tensors: Record<string, ort.Tensor>) => { for (const tensor of Object.values(tensors)) tensor.dispose() }

export class Model {
  private constructor(readonly session: ort.InferenceSession, readonly window: Float32Array, readonly basis: Float32Array) {}

  static async create(bytes: Uint8Array, window: Float32Array, basis: Float32Array) {
    if (window.length !== 400 || basis.length !== 40 * 201) throw new Error('Invalid CWformer mel assets.')
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' })
    if (session.inputNames.length !== 40 || session.outputNames.length !== 40) {
      await session.release()
      throw new Error('The CWformer model has an incompatible streaming layout.')
    }
    return new Model(session, window, basis)
  }

  state(): Record<string, ort.Tensor> {
    const state: Record<string, ort.Tensor> = {
      pos_offset: new ort.Tensor('int64', new BigInt64Array(1), [1]),
      sub_buf1: zeros([1, 1, 2, 40]),
      sub_buf2: zeros([1, 256, 2, 20]),
    }
    for (let layer = 0; layer < 12; layer++) {
      state[`kv_k_layer${layer}`] = zeros([1, 4, 0, 64])
      state[`kv_v_layer${layer}`] = zeros([1, 4, 0, 64])
      state[`conv_buf_layer${layer}`] = zeros([1, 256, 62])
    }
    return state
  }
}

// The 250-frame cache tail is per attention head, not one flat suffix.
export function cacheTail(tensor: ort.Tensor) {
  const length = tensor.dims[2]
  if (length <= 250) return tensor
  const source = tensor.data as Float32Array
  const data = new Float32Array(4 * 250 * 64)
  for (let head = 0; head < 4; head++) {
    const start = (head * length + length - 250) * 64
    data.set(source.subarray(start, start + 250 * 64), head * 250 * 64)
  }
  tensor.dispose()
  return new ort.Tensor('float32', data, [1, 4, 250, 64])
}

export class StreamingDecoder {
  private pending: Float32Array = new Float32Array(0)
  private previousToken = 0
  private emitted = false
  private trailingSpace = false
  private finished = false
  private quietFrames = 0
  private frontend!: Frontend
  private mel!: MelFrontend
  private state: Record<string, ort.Tensor> = {}

  private constructor(readonly model: Model, public settings: Settings) {}

  static async create(model: Model, settings: Settings) {
    validateSettings(settings)
    const decoder = new StreamingDecoder(model, { ...settings })
    try { await decoder.resetSignal(); return decoder }
    catch (error) { decoder.dispose(); throw error }
  }

  dispose() {
    release(this.state)
    this.state = {}
    this.pending = new Float32Array(0)
    this.finished = true
  }

  private async resetSignal() {
    this.frontend = new Frontend(this.settings.frequency, this.settings.bandwidth)
    this.mel = new MelFrontend(this.model.window, this.model.basis)
    release(this.state)
    this.state = this.model.state()
    this.quietFrames = 0
    this.previousToken = 0
    await this.infer(new Float32Array(16000))
  }

  private async retune() {
    validateSettings(this.settings)
    const retuned = Math.abs(this.frontend.frequency - this.settings.frequency) > Math.max(12, this.settings.bandwidth / 3)
    if (retuned || this.frontend.bandwidth !== this.settings.bandwidth) {
      this.pending = new Float32Array(0)
      this.trailingSpace = this.emitted
      await this.resetSignal()
    }
  }

  private async chunk(audio: Float32Array) {
    this.frontend.frequency = this.settings.frequency
    return this.infer(this.frontend.process(audio))
  }

  private async infer(audio: Float32Array) {
    const mel = this.mel.process(audio)
    if (!mel.frames) return ''
    const input = new ort.Tensor('float32', mel.data, [1, mel.frames, 40])
    let output: ort.InferenceSession.OnnxValueMapType
    try { output = await this.model.session.run({ mel_chunk: input, ...this.state }) }
    finally { input.dispose() }
    const tensors = this.model.session.outputNames.map(name => output[name] as ort.Tensor)
    const next: Record<string, ort.Tensor> = { pos_offset: tensors[1] }
    let at = 2
    for (let layer = 0; layer < 12; layer++) {
      next[`kv_k_layer${layer}`] = cacheTail(tensors[at++])
      next[`kv_v_layer${layer}`] = cacheTail(tensors[at++])
    }
    for (let layer = 0; layer < 12; layer++) next[`conv_buf_layer${layer}`] = tensors[at++]
    next.sub_buf1 = tensors[at++]
    next.sub_buf2 = tensors[at]
    release(this.state)
    this.state = next

    const logits = tensors[0]
    if (logits.dims.length !== 3 || logits.dims[1] !== 1 || logits.dims[2] !== TOKENS.length) {
      logits.dispose()
      throw new Error('The CWformer model has an incompatible vocabulary.')
    }
    const scores = logits.data as Float32Array
    let text = ''
    for (let frame = 0; frame < logits.dims[0]; frame++) {
      const offset = frame * TOKENS.length
      let token = 0
      for (let i = 1; i < TOKENS.length; i++) if (scores[offset + i] > scores[offset + token]) token = i
      this.quietFrames = token > 1 ? 0 : this.quietFrames + 1
      if (token && token !== this.previousToken) {
        if (token === 1) this.trailingSpace = this.emitted
        else {
          text += (this.trailingSpace ? ' ' : '') + TOKENS[token]
          this.emitted = true
          this.trailingSpace = false
        }
      }
      this.previousToken = token
    }
    logits.dispose()
    if (this.emitted && this.quietFrames >= 250) {
      release(this.state)
      this.state = this.model.state()
      this.previousToken = 0
      this.quietFrames = 0
    }
    return text
  }

  async feed(audio: Float32Array, progress?: (processed: number) => void) {
    if (this.finished) throw new Error('Cannot feed a finished decoder.')
    if (!(audio instanceof Float32Array) || audio.some(sample => !Number.isFinite(sample))) throw new Error('Audio must contain finite mono samples.')
    await this.retune()
    let text = ''
    let offset = 0
    if (this.pending.length) {
      const needed = 4000 - this.pending.length
      const take = Math.min(needed, audio.length)
      const joined = new Float32Array(this.pending.length + take)
      joined.set(this.pending)
      joined.set(audio.subarray(0, take), this.pending.length)
      this.pending = joined
      offset = take
      if (this.pending.length < 4000) return ''
      text += await this.chunk(this.pending)
      this.pending = new Float32Array(0)
      progress?.(offset)
    }
    for (; offset + 4000 <= audio.length; offset += 4000) {
      text += await this.chunk(audio.subarray(offset, offset + 4000))
      progress?.(offset + 4000)
    }
    this.pending = audio.slice(offset)
    return text
  }

  async flush() {
    if (this.finished) return ''
    await this.retune()
    let text = ''
    if (this.pending.length) {
      const padded = new Float32Array(4000)
      padded.set(this.pending)
      text += await this.chunk(padded)
    }
    text += await this.chunk(new Float32Array(4000))
    this.pending = new Float32Array(0)
    this.finished = true
    return text
  }
}
