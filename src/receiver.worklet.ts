import { CWDecoder, type ReceiverSettings } from './decoder'

declare const sampleRate: number
declare abstract class AudioWorkletProcessor {
  readonly port: MessagePort
  abstract process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean
}
declare function registerProcessor(name: string, processor: typeof AudioWorkletProcessor): void

class ReceiverProcessor extends AudioWorkletProcessor {
  private decoder = new CWDecoder(8000)
  private clock = 0
  private sum = 0
  private count = 0
  private buffer = new Float32Array(160)
  private index = 0
  private blocks = 0
  private active = true
  private capture = false

  constructor() {
    super()
    this.port.onmessage = ({ data }: MessageEvent<{ type: string; settings?: ReceiverSettings; enabled?: boolean }>) => {
      if (data.type === 'capture') this.capture = !!data.enabled
      if (data.type === 'configure') this.decoder.configure(data.settings ?? {})
      if (data.type === 'reset') this.decoder = new CWDecoder(8000, data.settings)
      if (data.type === 'stop') {
        if (this.index) {
          const samples = this.buffer.slice(0, this.index)
          if (this.capture) this.port.postMessage({ type: 'samples', samples })
          this.decoder.process(samples)
        }
        this.port.postMessage({ type: 'finished', reading: this.decoder.finish() })
        this.active = false
      }
    }
  }

  process(inputs: Float32Array[][]) {
    if (!this.active) return false
    const input = inputs[0]?.[0]
    if (!input) return true
    for (const sample of input) {
      this.sum += sample
      this.count++
      this.clock += 8000
      if (this.clock >= sampleRate) {
        this.clock -= sampleRate
        this.buffer[this.index++] = this.sum / this.count
        this.sum = 0; this.count = 0
        if (this.index === this.buffer.length) {
          if (this.capture) this.port.postMessage({ type: 'samples', samples: this.buffer.slice() })
          const reading = this.decoder.process(this.buffer)
          this.index = 0
          if (++this.blocks % 3 === 0) this.port.postMessage(reading)
        }
      }
    }
    return true
  }
}

registerProcessor('cw-receiver', ReceiverProcessor)
