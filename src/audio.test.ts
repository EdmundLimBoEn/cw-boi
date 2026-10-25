import { afterEach, expect, mock, test } from 'bun:test'
import { DEFAULT_SENDER, transmission } from './morse'

mock.module('./receiver.worklet.ts?worker&url', () => ({ default: 'receiver-worklet.js' }))
import { DEFAULT_RECEIVER } from './decoder'
const { StationAudio } = await import('./audio')

function deferred() {
  let resolve!: () => void
  let reject!: (error: Error) => void
  const promise = new Promise<void>((done, fail) => { resolve = done; reject = fail })
  return { promise, resolve, reject }
}

class Node {
  gain = { value: 0, setValueAtTime() {}, linearRampToValueAtTime() {}, setTargetAtTime() {}, cancelScheduledValues() {} }
  frequency = { value: 0 }
  Q = { value: 0 }
  onended: (() => void) | null = null
  stops = 0
  disconnected = false
  connect(node: Node) { return node }
  disconnect() { this.disconnected = true }
  start() {}
  stop() { this.stops++ }
}
const originalWindow = globalThis.window
const originalContext = globalThis.AudioContext
const originalWorklet = globalThis.AudioWorkletNode
const originalNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator')
afterEach(() => {
  globalThis.window = originalWindow
  globalThis.AudioContext = originalContext
  globalThis.AudioWorkletNode = originalWorklet
  if (originalNavigator) Object.defineProperty(globalThis, 'navigator', originalNavigator)
  else Reflect.deleteProperty(globalThis, 'navigator')
})

function setup() {
  const resumes: ReturnType<typeof deferred>[] = []
  const sources: Node[] = []
  const copyToChannel = mock((_samples: Float32Array, _channel: number) => {})
  const module = deferred()
  class Context {
    currentTime = 0
    destination = new Node()
    audioWorklet = { addModule: () => module.promise }
    resume() { const pending = deferred(); resumes.push(pending); return pending.promise }
    createGain() { return new Node() }
    createBiquadFilter() { return new Node() }
    createMediaStreamSource() { return new Node() }
    createOscillator() { const node = new Node(); sources.push(node); return node }
    createBufferSource() { return this.createOscillator() }
    createBuffer() { return { copyToChannel } }
  }
  globalThis.window = { isSecureContext: true } as Window & typeof globalThis
  globalThis.AudioContext = Context as unknown as typeof AudioContext
  return { audio: new StationAudio(), resumes, sources, module, copyToChannel }
}
const starts = {
  tones: (audio: InstanceType<typeof StationAudio>, onEnd: () => void) => audio.play(transmission('CQ', 20), { frequency: 650, volume: 0.5, wpm: 20, spacing: 20 }, false, onEnd),
  samples: (audio: InstanceType<typeof StationAudio>, onEnd: () => void) => audio.playSamples(new Float32Array(800), 8000, 0.5, onEnd),
}

test('sample playback passes the original sample view to copyToChannel without cloning', async () => {
  const { audio, resumes, module, copyToChannel } = setup()
  const samples = new Float32Array([1, 0.25, -0.5, 1]).subarray(1, 3)
  module.resolve()
  const pending = audio.playSamples(samples, 8000, 0.5, () => {})
  resumes[0].resolve()
  expect(await pending).toBe(true)
  expect(copyToChannel).toHaveBeenCalledTimes(1)
  expect(copyToChannel.mock.calls[0][0]).toBe(samples)
  expect(copyToChannel.mock.calls[0][1]).toBe(0)
})

for (const [kind, start] of Object.entries(starts)) {
  test(`${kind}: stop while loading the worklet prevents source creation`, async () => {
    const { audio, resumes, sources, module } = setup()
    const ended = mock(() => {})
    const pending = start(audio, ended)
    resumes[0].resolve()
    await Promise.resolve()
    audio.stopSender()
    module.resolve()
    await pending
    expect(sources).toHaveLength(0)
    expect(ended).not.toHaveBeenCalled()
  })
  for (const reverse of [false, true]) {
    test(`${kind}: cancelled readiness cannot replace newer playback (${reverse ? 'new' : 'old'} resumes first)`, async () => {
      const { audio, resumes, sources, module } = setup()
      module.resolve()
      const oldEnd = mock(() => {}), newEnd = mock(() => {})
      const old = start(audio, oldEnd)
      audio.stopSender()
      const current = start(audio, newEnd)
      resumes[reverse ? 1 : 0].resolve()
      await (reverse ? current : old)
      resumes[reverse ? 0 : 1].resolve()
      await Promise.all([old, current])
      expect(sources).toHaveLength(1)
      expect(sources[0].disconnected).toBe(false)
      expect(oldEnd).not.toHaveBeenCalled()
      sources[0].onended!()
      expect(newEnd).toHaveBeenCalledTimes(1)
      expect(sources[0].disconnected).toBe(true)
    })
  }
  test(`${kind}: replacing an active source disconnects it and ignores its queued end event`, async () => {
    const { audio, resumes, sources, module } = setup()
    module.resolve()
    const oldEnd = mock(() => {}), newEnd = mock(() => {})
    const old = start(audio, oldEnd)
    resumes[0].resolve(); await old
    const queuedEnd = sources[0].onended!
    const current = start(audio, newEnd)
    resumes[1].resolve(); await current
    queuedEnd()
    expect(sources[0].disconnected).toBe(true)
    expect(sources[1].disconnected).toBe(false)
    expect(oldEnd).not.toHaveBeenCalled()
    expect(newEnd).not.toHaveBeenCalled()
    audio.stopSender()
    expect(sources[1].disconnected).toBe(true)
  })
}

test('straight key: stopping pending setup prevents oscillator creation', async () => {
  const { audio, resumes, sources, module } = setup()
  const pending = audio.prepareKey(DEFAULT_SENDER)
  resumes[0].resolve()
  await Promise.resolve()
  audio.stopKey()
  module.resolve()
  await pending
  expect(sources).toHaveLength(0)
})

for (const reverse of [false, true]) {
  test(`straight key: cancelled setup cannot replace the new key (${reverse ? 'new' : 'old'} resumes first)`, async () => {
    const { audio, resumes, sources, module } = setup()
    module.resolve()
    const old = audio.prepareKey(DEFAULT_SENDER)
    audio.stopKey()
    const current = audio.prepareKey(DEFAULT_SENDER)
    resumes[reverse ? 1 : 0].resolve()
    await (reverse ? current : old)
    resumes[reverse ? 0 : 1].resolve()
    await Promise.all([old, current])
    expect(sources).toHaveLength(1)
    expect(sources[0].stops).toBe(0)
    expect(sources[0].disconnected).toBe(false)
    audio.stopKey()
    expect(sources[0].stops).toBe(1)
    expect(sources[0].disconnected).toBe(true)
  })
}

async function microphone() {
  const state = setup()
  const worklets: Worklet[] = []
  class Worklet extends Node {
    messages: { type: string }[] = []
    port = {
      onmessage: null as null | ((event: { data: unknown }) => void),
      postMessage: (message: { type: string }) => { this.messages.push(message) },
      close: mock(() => {}),
    }
    constructor() { super(); worklets.push(this) }
    emit(data: unknown) { this.port.onmessage?.({ data }) }
  }
  const track = { onended: null as null | (() => Promise<void>), stop: mock(() => {}) }
  const stream = { getTracks: () => [track], getAudioTracks: () => [track] }
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { mediaDevices: { getUserMedia: async () => stream } } })
  globalThis.AudioWorkletNode = Worklet as unknown as typeof AudioWorkletNode
  state.module.resolve()
  const pending = state.audio.listen(DEFAULT_RECEIVER)
  state.resumes[0].resolve()
  // Wait until listen has acquired the stream and entered createReceiver's readiness.
  while (state.resumes.length < 2) await Promise.resolve()
  state.resumes[1].resolve()
  expect(await pending).toBe(true)
  return { ...state, track, worklets }
}

test('microphone disconnect delivers buffered samples and final reading before ending reception', async () => {
  const { audio, track, worklets } = await microphone()
  const events: string[] = []
  const tail = new Float32Array([0.2, 0.1])
  audio.onSamples = samples => { expect(samples).toEqual(tail); events.push('samples') }
  audio.onReading = reading => { expect(reading.text).toBe('E'); events.push('reading') }
  audio.onReceiverEnded = () => { events.push('ended') }
  const pending = track.onended!()
  expect(worklets[0].messages.at(-1)).toEqual({ type: 'stop' })
  expect(worklets[0].port.close).not.toHaveBeenCalled()
  expect(events).toEqual([])
  worklets[0].emit({ type: 'samples', samples: tail })
  worklets[0].emit({ type: 'finished', reading: { text: 'E' } })
  await pending
  expect(events).toEqual(['samples', 'reading', 'ended'])
  expect(worklets[0].port.close).toHaveBeenCalledTimes(1)
  expect(track.stop).toHaveBeenCalledTimes(1)
})

test('microphone disconnect completion cannot affect a newer receiver', async () => {
  const { audio, track, worklets, resumes } = await microphone()
  const ended = mock(() => {}), reading = mock(() => {}), samples = mock(() => {})
  audio.onReceiverEnded = ended
  audio.onReading = reading
  audio.onSamples = samples
  const queuedEnd = track.onended!
  const pending = queuedEnd()
  const current = audio.loopback(DEFAULT_RECEIVER)
  resumes[2].resolve()
  expect(await current).toBe(true)
  worklets[0].emit({ type: 'samples', samples: new Float32Array([1]) })
  worklets[0].emit({ type: 'finished', reading: { text: 'OLD' } })
  await pending
  await queuedEnd()
  expect(ended).not.toHaveBeenCalled()
  expect(reading).not.toHaveBeenCalled()
  expect(samples).not.toHaveBeenCalled()
  expect(worklets[1].disconnected).toBe(false)
  expect(worklets[1].messages.some(message => message.type === 'stop')).toBe(false)
  await audio.stopReceiver()
})

for (const disconnected of [false, true]) {
  test(`overlapping graceful stops preserve the flush (${disconnected ? 'microphone ended' : 'explicit stop'})`, async () => {
    const { audio, track, worklets } = await microphone()
    const samples = mock(() => {}), reading = mock(() => {}), ended = mock(() => {})
    audio.onSamples = samples
    audio.onReading = reading
    audio.onReceiverEnded = ended
    const first = disconnected ? track.onended!() : audio.stopReceiver(true)
    let secondFinished = false
    const second = audio.stopReceiver(true).then(() => { secondFinished = true })
    await Promise.resolve()
    expect(secondFinished).toBe(false)
    const tail = new Float32Array([0.2, 0.1])
    worklets[0].emit({ type: 'samples', samples: tail })
    worklets[0].emit({ type: 'finished', reading: { text: 'E' } })
    await Promise.all([first, second])
    expect(samples).toHaveBeenCalledTimes(1)
    expect(samples).toHaveBeenCalledWith(tail)
    expect(reading).toHaveBeenCalledTimes(1)
    expect(reading).toHaveBeenCalledWith({ text: 'E' })
    expect(ended).toHaveBeenCalledTimes(disconnected ? 1 : 0)
    expect(worklets[0].messages.filter(message => message.type === 'stop')).toHaveLength(1)
    expect(worklets[0].port.close).toHaveBeenCalledTimes(1)
    expect(track.stop).toHaveBeenCalledTimes(1)
  })
}

test('explicit cancellation invalidates a pending graceful flush', async () => {
  const { audio, track, worklets } = await microphone()
  const samples = mock(() => {}), reading = mock(() => {}), ended = mock(() => {})
  audio.onSamples = samples
  audio.onReading = reading
  audio.onReceiverEnded = ended
  const pending = track.onended!()
  await audio.stopReceiver()
  worklets[0].emit({ type: 'samples', samples: new Float32Array([1]) })
  worklets[0].emit({ type: 'finished', reading: { text: 'OLD' } })
  await pending
  expect(samples).not.toHaveBeenCalled()
  expect(reading).not.toHaveBeenCalled()
  expect(ended).not.toHaveBeenCalled()
  expect(worklets[0].port.close).toHaveBeenCalledTimes(1)
})

test('stale microphone setup failure stops its tracks without disconnecting a replacement', async () => {
  const { audio, resumes, worklets } = await microphone()
  const oldTrack = { stop: mock(() => {}) }
  const acquired = deferred()
  navigator.mediaDevices.getUserMedia = async () => {
    acquired.resolve()
    return { getTracks: () => [oldTrack] } as unknown as MediaStream
  }
  const old = audio.listen(DEFAULT_RECEIVER)
  resumes[2].resolve()
  await acquired.promise
  await new Promise(resolve => setTimeout(resolve, 0))
  expect(resumes).toHaveLength(4)
  await audio.stopReceiver()
  const current = audio.loopback(DEFAULT_RECEIVER)
  resumes[4].resolve()
  expect(await current).toBe(true)
  const failure = new Error('old readiness failed')
  resumes[3].reject(failure)
  await expect(old).rejects.toThrow(failure.message)
  expect(oldTrack.stop).toHaveBeenCalledTimes(1)
  expect(worklets[1].disconnected).toBe(false)
  expect(worklets[1].port.close).not.toHaveBeenCalled()
  audio.configure(DEFAULT_RECEIVER)
  expect(worklets[1].messages.at(-1)).toEqual({ type: 'configure', settings: DEFAULT_RECEIVER })
  await audio.stopReceiver()
})
