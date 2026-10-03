import type { ReceiverSettings } from './decoder'

export type NeuralEngine = 'rnnt' | 'cwformer'
export type NeuralEngineInfo = { engine: NeuralEngine; model: string; bandwidth: number }

async function request(path: string, body?: Float32Array, signal?: AbortSignal) {
  const response = await fetch(`/api/${path}`, { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: body ? Float32Array.from(body).buffer : undefined, signal })
  const data = await response.json()
  if (!response.ok) throw new Error(data.error || 'The local neural engine is unavailable.')
  return data
}

const parameters = (settings: ReceiverSettings, engine: NeuralEngine) => new URLSearchParams({ engine, frequency: String(settings.frequency), bandwidth: String(settings.bandwidth) }).toString()

export async function neuralDecode(samples: Float32Array, settings: ReceiverSettings, engine: NeuralEngine, signal?: AbortSignal): Promise<string> {
  return (await request(`decode?${parameters(settings, engine)}`, samples, signal)).text
}

export class NeuralStream {
  constructor(readonly engine: NeuralEngine = 'rnnt') {}
  frequency = 650
  bandwidth = 100
  private session = ''
  private queue: Float32Array[] = []
  private pumping: Promise<void> | null = null
  private finalizing: Promise<void> | null = null
  private cancelled = false
  private finished = false
  onText: (text: string) => void = () => {}
  onError: (error: unknown) => void = () => {}

  async start(settings: ReceiverSettings) {
    this.frequency = settings.frequency; this.bandwidth = settings.bandwidth
    const data = await request(`stream/start?${parameters(settings, this.engine)}`)
    this.session = data.session
    if (this.cancelled) this.cancel()
  }

  feed(samples: Float32Array) {
    if (this.cancelled || this.finished || !this.session) return
    if (this.queue.length >= 600) { this.cancel(); this.onError(new Error('The neural engine cannot keep up. Use the adaptive decoder or a faster inference device.')); return }
    this.queue.push(samples)
    if (this.queue.reduce((count, chunk) => count + chunk.length, 0) >= 8000) this.pumping ??= this.pump()
  }

  private async pump() {
    try {
      while (this.queue.length && !this.cancelled) {
        const chunks = this.queue.splice(0)
        const merged = new Float32Array(chunks.reduce((count, chunk) => count + chunk.length, 0))
        let offset = 0
        for (const chunk of chunks) { merged.set(chunk, offset); offset += chunk.length }
        const data = await request(`stream/feed?session=${this.session}&engine=${this.engine}&frequency=${this.frequency}&bandwidth=${this.bandwidth}`, merged)
        if (!this.cancelled) this.onText(data.text)
      }
    } catch (error) { if (!this.cancelled) { this.cancel(); this.onError(error) } }
    finally { this.pumping = null }
  }

  finish() {
    return this.finalizing ??= this.complete()
  }

  private async complete() {
    this.finished = true
    if (!this.pumping && this.queue.length) this.pumping = this.pump()
    await this.pumping
    if (this.cancelled || !this.session) return
    try {
      const data = await request(`stream/finish?session=${this.session}&engine=${this.engine}`)
      if (!this.cancelled) this.onText(data.text)
      this.session = ''
    } catch (error) { if (!this.cancelled) this.onError(error) }
  }

  cancel() {
    this.cancelled = true; this.queue = []
    if (this.session) void request(`stream/cancel?session=${this.session}&engine=${this.engine}`).catch(() => {})
    this.session = ''
  }
}
