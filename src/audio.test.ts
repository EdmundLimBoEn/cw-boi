import { afterEach, expect, mock, test } from 'bun:test'
import { transmission } from './morse'

mock.module('./receiver.worklet.ts?worker&url', () => ({ default: 'receiver-worklet.js' }))
const { StationAudio } = await import('./audio')

function deferred() {
  let resolve!: () => void
  const promise = new Promise<void>(done => { resolve = done })
  return { promise, resolve }
}

class Node {
  gain = { value: 0, setValueAtTime() {}, linearRampToValueAtTime() {} }
  frequency = { value: 0 }
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
afterEach(() => {
  globalThis.window = originalWindow
  globalThis.AudioContext = originalContext
})

function setup() {
  const resumes: ReturnType<typeof deferred>[] = []
  const sources: Node[] = []
  const module = deferred()
  class Context {
    currentTime = 0
    destination = new Node()
    audioWorklet = { addModule: () => module.promise }
    resume() { const pending = deferred(); resumes.push(pending); return pending.promise }
    createGain() { return new Node() }
    createOscillator() { const node = new Node(); sources.push(node); return node }
    createBufferSource() { return this.createOscillator() }
    createBuffer() { return { copyToChannel() {} } }
  }
  globalThis.window = { isSecureContext: true } as Window & typeof globalThis
  globalThis.AudioContext = Context as unknown as typeof AudioContext
  return { audio: new StationAudio(), resumes, sources, module }
}
const starts = {
  tones: (audio: InstanceType<typeof StationAudio>, onEnd: () => void) => audio.play(transmission('CQ', 20), { frequency: 650, volume: 0.5, wpm: 20, spacing: 20 }, false, onEnd),
  samples: (audio: InstanceType<typeof StationAudio>, onEnd: () => void) => audio.playSamples(new Float32Array(800), 8000, 0.5, onEnd),
}

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
