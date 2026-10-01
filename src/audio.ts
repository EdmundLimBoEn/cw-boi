import processorUrl from './receiver.worklet.ts?worker&url'
import type { DecoderReading, ReceiverSettings } from './decoder'
import type { SenderSettings, Transmission } from './morse'

export class StationAudio {
  context: AudioContext | null = null
  private loaded = false
  private receiver: AudioWorkletNode | null = null
  private input: AudioNode | null = null
  private filter: BiquadFilterNode | null = null
  private silence: GainNode | null = null
  private stream: MediaStream | null = null
  private player: AudioScheduledSourceNode | null = null
  private txGain: GainNode | null = null
  private outputGain: GainNode | null = null
  private keyOscillator: OscillatorNode | null = null
  private keyGain: GainNode | null = null
  private keyOutput: GainNode | null = null
  private generation = 0
  private loadPromise: Promise<void> | null = null
  onReading: (reading: DecoderReading) => void = () => {}
  onReceiverEnded: () => void = () => {}
  onSamples: ((samples: Float32Array) => void) | null = null

  async ready() {
    if (!window.isSecureContext) throw new Error('Audio needs HTTPS or localhost. Open this station through a secure connection.')
    this.context ??= new AudioContext({ latencyHint: 'interactive' })
    await this.context.resume()
    if (!this.loaded) {
      this.loadPromise ??= this.context.audioWorklet.addModule(processorUrl)
      try { await this.loadPromise } catch (error) { this.loadPromise = null; throw error }
      this.loaded = true
    }
    return this.context
  }

  private async createReceiver(settings: ReceiverSettings) {
    const generation = this.generation
    const context = await this.ready()
    if (generation !== this.generation) return false
    this.receiver = new AudioWorkletNode(context, 'cw-receiver', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1], channelCount: 1, channelCountMode: 'explicit' })
    this.receiver.port.postMessage({ type: 'reset', settings })
    const capture = this.onSamples
    this.receiver.port.onmessage = ({ data }) => data.type === 'samples' ? capture?.(data.samples) : this.onReading(data)
    this.receiver.port.postMessage({ type: 'capture', enabled: !!this.onSamples })
    this.filter = context.createBiquadFilter()
    this.filter.type = 'lowpass'
    this.filter.frequency.value = 1700
    this.filter.Q.value = 0.707
    this.filter.connect(this.receiver)
    this.silence = context.createGain()
    this.silence.gain.value = 0
    this.receiver.connect(this.silence).connect(context.destination)
    return true
  }

  async listen(settings: ReceiverSettings, deviceId?: string) {
    this.stopReceiver()
    const generation = this.generation
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('This browser cannot access audio input. Try an audio file or a current desktop browser.')
    const context = await this.ready()
    if (generation !== this.generation) return false
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { deviceId: deviceId ? { exact: deviceId } : undefined, echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 } })
    if (generation !== this.generation) { stream.getTracks().forEach(track => track.stop()); return false }
    try {
      const created = await this.createReceiver(settings)
      if (!created || generation !== this.generation) { stream.getTracks().forEach(track => track.stop()); return false }
      this.stream = stream
      this.input = context.createMediaStreamSource(stream)
      this.input.connect(this.filter!)
      stream.getAudioTracks().forEach(track => { track.onended = () => { this.stopReceiver(); this.onReceiverEnded() } })
      return true
    } catch (error) {
      stream.getTracks().forEach(track => track.stop())
      this.stopReceiver()
      throw error
    }
  }

  async loopback(settings: ReceiverSettings) {
    this.stopReceiver()
    return this.createReceiver(settings)
  }

  configure(settings: ReceiverSettings) {
    this.receiver?.port.postMessage({ type: 'configure', settings })
  }

  clear(settings: ReceiverSettings) {
    this.receiver?.port.postMessage({ type: 'reset', settings })
  }

  async stopReceiver(flush = false) {
    const generation = ++this.generation
    this.input?.disconnect()
    this.input = null
    this.stream?.getTracks().forEach(track => { track.onended = null; track.stop() })
    this.stream = null
    const receiver = this.receiver, filter = this.filter, silence = this.silence, capture = this.onSamples
    this.receiver = null
    this.filter = null; this.silence = null
    filter?.disconnect()
    if (receiver) {
      if (flush) await new Promise<void>(resolve => {
        const timer = setTimeout(resolve, 300)
        receiver.port.onmessage = ({ data }) => {
          if (generation === this.generation) {
            if (data.type === 'samples') capture?.(data.samples)
            else this.onReading(data.reading ?? data)
          }
          if (data.type === 'finished') { clearTimeout(timer); resolve() }
        }
        receiver.port.postMessage({ type: 'stop' })
      })
      receiver.port.onmessage = null
      receiver.disconnect(); receiver.port.close()
    }
    silence?.disconnect()
  }

  async play(plan: Transmission, settings: SenderSettings, receive: boolean, onEnd: () => void) {
    this.stopSender()
    const context = await this.ready()
    const oscillator = context.createOscillator()
    oscillator.frequency.value = settings.frequency
    const gain = context.createGain()
    const output = context.createGain()
    output.gain.value = settings.volume
    oscillator.connect(gain).connect(output).connect(context.destination)
    if (receive && this.filter) gain.connect(this.filter)
    const start = context.currentTime + 0.06
    gain.gain.value = 0
    for (const tone of plan.tones) {
      gain.gain.setValueAtTime(0, start + tone.start)
      gain.gain.linearRampToValueAtTime(0.6, start + tone.start + 0.004)
      gain.gain.setValueAtTime(0.6, start + tone.end - 0.004)
      gain.gain.linearRampToValueAtTime(0, start + tone.end)
    }
    oscillator.start(start)
    oscillator.stop(start + plan.duration)
    oscillator.onended = () => {
      if (this.player !== oscillator) return
      this.stopSender()
      onEnd()
    }
    this.player = oscillator; this.txGain = gain; this.outputGain = output
    return start
  }

  async playSamples(samples: Float32Array, sampleRate: number, volume: number, onEnd: () => void) {
    this.stopSender()
    const context = await this.ready()
    const buffer = context.createBuffer(1, samples.length, sampleRate)
    buffer.copyToChannel(Float32Array.from(samples), 0)
    const source = context.createBufferSource()
    const output = context.createGain()
    output.gain.value = volume
    source.buffer = buffer
    source.connect(output).connect(context.destination)
    if (this.filter) source.connect(this.filter)
    source.onended = () => {
      if (this.player !== source) return
      this.stopSender()
      onEnd()
    }
    this.player = source; this.outputGain = output
    source.start()
  }

  setVolume(volume: number) {
    if (this.context) this.outputGain?.gain.setTargetAtTime(volume, this.context.currentTime, 0.015)
    if (this.context) this.keyOutput?.gain.setTargetAtTime(volume, this.context.currentTime, 0.015)
  }

  stopSender() {
    if (this.player) {
      this.player.onended = null
      try { this.player.stop() } catch { /* The source may already have finished. */ }
      this.player.disconnect()
    }
    this.player = null
    this.txGain?.disconnect(); this.txGain = null
    this.outputGain?.disconnect(); this.outputGain = null
  }

  async prepareKey(settings: SenderSettings) {
    this.releaseKey()
    const context = await this.ready()
    this.keyOscillator?.stop(); this.keyOscillator?.disconnect()
    this.keyGain?.disconnect(); this.keyOutput?.disconnect()
    this.keyOscillator = context.createOscillator()
    this.keyOscillator.frequency.value = settings.frequency
    this.keyGain = context.createGain()
    this.keyGain.gain.value = 0
    this.keyOutput = context.createGain()
    this.keyOutput.gain.value = settings.volume
    this.keyOscillator.connect(this.keyGain).connect(this.keyOutput).connect(context.destination)
    if (this.filter) this.keyGain.connect(this.filter)
    this.keyOscillator.start()
  }

  pressKey() {
    if (!this.context || !this.keyGain) return
    this.keyGain.gain.cancelScheduledValues(this.context.currentTime)
    this.keyGain.gain.setTargetAtTime(0.6, this.context.currentTime, 0.002)
  }

  releaseKey() {
    if (!this.context || !this.keyGain) return
    this.keyGain.gain.cancelScheduledValues(this.context.currentTime)
    this.keyGain.gain.setTargetAtTime(0, this.context.currentTime, 0.002)
  }

  stopKey() {
    this.releaseKey()
    this.keyOscillator?.stop(); this.keyOscillator?.disconnect(); this.keyOscillator = null
    this.keyGain?.disconnect(); this.keyGain = null
    this.keyOutput?.disconnect(); this.keyOutput = null
  }

  async readFile(file: File) {
    if (file.size > 50 * 1024 * 1024) throw new Error('Choose an audio file smaller than 50 MB.')
    const context = await this.ready()
    const buffer = await context.decodeAudioData(await file.arrayBuffer())
    if (buffer.duration > 600) throw new Error('Choose a recording shorter than 10 minutes.')
    const offline = new OfflineAudioContext(1, Math.ceil(buffer.duration * 8000), 8000)
    const source = offline.createBufferSource()
    source.buffer = buffer
    source.connect(offline.destination)
    source.start()
    const rendered = await offline.startRendering()
    return rendered.getChannelData(0)
  }

  dispose() {
    this.stopSender(); this.stopReceiver(); this.stopKey()
    void this.context?.close()
    this.context = null
  }
}

export function audioError(error: unknown) {
  if (error instanceof DOMException) {
    if (error.name === 'NotAllowedError') return 'Microphone access is blocked. Allow it in your browser’s site settings, then try again.'
    if (error.name === 'NotFoundError') return 'No audio input found. Connect a microphone or choose an audio file.'
    if (error.name === 'NotReadableError') return 'This audio input is unavailable. Close other apps using it or choose another input.'
    if (error.name === 'EncodingError') return 'This file could not be read. Try a WAV, MP3, OGG, or M4A recording supported by your browser.'
  }
  return error instanceof Error ? error.message : 'Audio could not start. Please try again.'
}
