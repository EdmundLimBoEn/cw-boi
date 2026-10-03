import type { ReceiverSettings } from './decoder'
import type { WorkerRequest, WorkerResponse } from './cwformer/protocol'

export type NeuralEngine = 'cwformer'
export type ModelState = { phase: 'idle' | 'loading' | 'ready' | 'error'; stage?: 'download' | 'initialize'; loaded?: number; total?: number; message?: string }
type Pending = { resolve: (text: string) => void; reject: (error: Error) => void; progress?: (value: number) => void }

export const NEURAL_ENGINE = { engine: 'cwformer' as const, model: 'CWformer v6 · on device', bandwidth: 150 }
let worker: Worker | null = null
let requestId = 0
let preparing: Promise<void> | null = null
let modelState: ModelState = { phase: 'idle' }
let activeStream: NeuralStream | null = null
const pending = new Map<number, Pending>()
const listeners = new Set<(state: ModelState) => void>()
const aborted = () => new DOMException('Decoding cancelled.', 'AbortError')

function publish(state: ModelState) {
  modelState = state
  for (const listener of listeners) listener(state)
}

export function subscribeModel(listener: (state: ModelState) => void) {
  listeners.add(listener); listener(modelState)
  return () => { listeners.delete(listener) }
}

function release(error: Error, state: ModelState) {
  worker?.terminate(); worker = null; preparing = null
  for (const request of pending.values()) request.reject(error)
  pending.clear()
  publish(state)
}

export function cancelNeural() {
  activeStream = null
  release(aborted(), { phase: 'idle' })
}

function getWorker() {
  if (worker) return worker
  const created = new Worker(new URL('./cwformer/worker.ts', import.meta.url), { type: 'module' })
  worker = created
  created.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
    if (worker !== created) return
    const request = pending.get(data.id)
    if (!request) return
    if (data.type === 'progress') {
      if (data.stage === 'decode') request.progress?.(data.total ? data.loaded / data.total : 0)
      else publish({ phase: 'loading', stage: data.stage, loaded: data.loaded, total: data.total })
    } else {
      pending.delete(data.id)
      if (data.type === 'error') request.reject(new Error(data.message))
      else request.resolve(data.text)
    }
  }
  created.onerror = event => {
    event.preventDefault()
    if (worker !== created) return
    const message = 'CWformer could not run on this browser. Retry or select the adaptive decoder.'
    release(new Error(message), { phase: 'error', message })
  }
  return created
}

function request(type: WorkerRequest['type'], frequency = 650, bandwidth = 150, samples?: Float32Array<ArrayBuffer>, progress?: (value: number) => void) {
  return new Promise<string>((resolve, reject) => {
    const target = getWorker(), id = ++requestId
    pending.set(id, { resolve, reject, progress })
    try { target.postMessage({ id, type, frequency, bandwidth, samples }, samples ? [samples.buffer] : []) }
    catch (error) { pending.delete(id); reject(error) }
  })
}

export function loadNeural(): Promise<void> {
  if (modelState.phase === 'ready') return Promise.resolve()
  if (preparing) return preparing
  publish({ phase: 'loading', stage: 'download', loaded: 0, total: 0 })
  let target: Worker
  try { target = getWorker() }
  catch (error) {
    const failure = error instanceof Error ? error : new Error('This browser does not support background decoding.')
    release(failure, { phase: 'error', message: failure.message })
    return Promise.reject(failure)
  }
  preparing = request('load').then(() => {
    if (worker !== target) throw aborted()
    preparing = null
    publish({ phase: 'ready' })
  }).catch(error => {
    if (worker === target) release(error, { phase: 'error', message: error instanceof Error ? error.message : 'The model could not load.' })
    throw error
  })
  return preparing
}

export class NeuralStream {
  constructor(readonly engine: NeuralEngine = 'cwformer') {}
  frequency = 650
  bandwidth = 150
  onText: (text: string) => void = () => {}
  onError: (error: unknown) => void = () => {}
  private queue: Float32Array[] = []
  private pendingSamples = 0
  private pumping: Promise<void> | null = null
  private finalizing: Promise<string> | null = null
  private cancelled = false
  private finished = false
  private started = false
  private text = ''

  async start(settings: ReceiverSettings) {
    if (this.cancelled) throw aborted()
    activeStream?.cancel()
    activeStream = this
    this.frequency = settings.frequency; this.bandwidth = settings.bandwidth
    await loadNeural()
    if (this.cancelled || activeStream !== this) throw aborted()
    await request('start', this.frequency, this.bandwidth)
    if (this.cancelled || activeStream !== this) throw aborted()
    this.started = true
  }

  feed(samples: Float32Array) {
    if (this.cancelled || this.finished || !this.started) return
    // At most 12 seconds of unprocessed input, including the in-flight chunk.
    if (this.pendingSamples + samples.length > 96000) {
      this.cancel()
      this.onError(new Error('This device cannot decode live CWformer audio fast enough. Select the adaptive decoder or decode a recording.'))
      return
    }
    this.pendingSamples += samples.length
    this.queue.push(samples.slice())
    if (this.pendingSamples >= 8000) this.pumping ??= this.pump()
  }

  private async pump() {
    try {
      while (this.queue.length && !this.cancelled) {
        const chunks = this.queue.splice(0)
        const samples = new Float32Array(chunks.reduce((length, chunk) => length + chunk.length, 0))
        let offset = 0
        for (const chunk of chunks) { samples.set(chunk, offset); offset += chunk.length }
        const text = await request('feed', this.frequency, this.bandwidth, samples)
        this.pendingSamples -= offset
        if (!this.cancelled) { this.text += text; this.onText(this.text) }
      }
    } catch (error) {
      if (!this.cancelled) { this.cancel(); this.onError(error) }
    } finally { this.pumping = null }
  }

  async decode(samples: Float32Array, progress?: (value: number) => void) {
    if (this.cancelled || !this.started) throw aborted()
    const text = await request('feed', this.frequency, this.bandwidth, samples.slice(), progress)
    if (this.cancelled) throw aborted()
    this.text += text; this.onText(this.text)
  }

  finish(): Promise<string> { return this.finalizing ??= this.complete() }

  private async complete() {
    this.finished = true
    if (!this.pumping && this.queue.length) this.pumping = this.pump()
    await this.pumping
    if (this.cancelled || !this.started) return this.text
    const text = await request('finish', this.frequency, this.bandwidth)
    if (!this.cancelled) {
      this.text += text; this.onText(this.text)
      if (activeStream === this) activeStream = null
    }
    return this.text
  }

  cancel() {
    this.cancelled = true; this.queue = []; this.pendingSamples = 0
    if (activeStream === this) {
      activeStream = null
      release(aborted(), modelState.phase === 'error' ? modelState : { phase: 'idle' })
    }
  }
}

export async function neuralDecode(samples: Float32Array, settings: ReceiverSettings, engine: NeuralEngine, signal?: AbortSignal, progress?: (value: number) => void) {
  if (signal?.aborted) throw aborted()
  const stream = new NeuralStream(engine)
  const cancel = () => stream.cancel()
  signal?.addEventListener('abort', cancel, { once: true })
  try {
    await stream.start(settings)
    await stream.decode(samples, progress)
    const text = await stream.finish()
    if (signal?.aborted) throw aborted()
    return text
  } catch (error) { stream.cancel(); throw error }
  finally { signal?.removeEventListener('abort', cancel) }
}
