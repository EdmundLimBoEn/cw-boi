import { createHash } from 'node:crypto'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { parseArgs } from 'node:util'
import { CHALLENGES, makeSignal, randomSource } from '../src/signals'
import { NeuralStream } from '../src/neural'

const { values } = parseArgs({ options: {
  seed: { type: 'string', default: '3401901' }, output: { type: 'string', default: '.research/acquisition-dev' },
  scanner: { type: 'string', default: 'src/decoder.ts' }, name: { type: 'string', default: 'candidate' },
  'noise-cases': { type: 'string', default: '12' },
}, strict: true })
const seed = Number(values.seed), noiseCases = Number(values['noise-cases'])
if (!/^\d+$/.test(values.seed!) || !Number.isSafeInteger(seed) || seed < 1 || seed > 0xffffffff) throw new Error('Seed must be a nonzero uint32')
if (!Number.isInteger(noiseCases) || noiseCases < 1 || noiseCases > 1000) throw new Error('Noise cases must be in [1, 1000]')
if (!/^[a-z0-9-]+$/.test(values.name!)) throw new Error('Name must contain lowercase letters, digits or hyphens')
const directory = resolve(values.output!), scannerPath = resolve(values.scanner!)
const { CWDecoder, DEFAULT_RECEIVER } = await import(pathToFileURL(scannerPath).href)
const rng = randomSource(seed), cases: Array<Record<string, any>> = []
const between = (a: number, b: number) => a + rng() * (b - a)
const sha256 = (bytes: Uint8Array | string) => createHash('sha256').update(bytes).digest('hex')
await mkdir(directory, { recursive: true })
function signal(text: string, frequency: number, wpm: number, snr: number) {
  return makeSignal(text, { ...CHALLENGES[0], frequency, wpm, snr }, Math.floor(rng() * 0x7fffffff) + 1)
}
async function save(kind: string, audio: Float32Array, expected: string, frequency: number, extra: Record<string, unknown> = {}) {
  const id = `${kind}-${cases.filter(item => item.condition === kind).length + 1}`, file = `${id}.f32`
  const bytes = Buffer.alloc(audio.length * 4)
  for (let i = 0; i < audio.length; i++) bytes.writeFloatLE(audio[i], i * 4)
  await writeFile(join(directory, file), bytes)
  cases.push({ id, condition: kind, file, sha256: sha256(bytes), expected, frequency, durationSeconds: audio.length / 8000, ...extra })
}
for (let i = 0; i < 16; i++) {
  const frequency = Math.round(between(400, 1150)), gain = 10 ** between(Math.log10(1.2e-5 / .4), Math.log10(8e-5 / .4))
  const expected = `CQ DE W${i % 10}ABC TEST 73`, body = signal(expected, frequency, Math.round(between(15, 30)), between(20, 35)).map(value => value * gain)
  const lead = Math.round(between(0, 2) * 8000), audio = new Float32Array(lead + body.length)
  audio.set(body, lead)
  await save('low-amplitude', audio, expected, frequency, { gain, onset: lead / 8000 })
}
for (let i = 0; i < 8; i++) {
  const frequency = Math.round(between(400, 1150)), expected = `CQ DE K${i}XYZ TEST 73`
  await save('normal-clean', signal(expected, frequency, Math.round(between(15, 30)), between(20, 35)), expected, frequency)
}
for (let i = 0; i < 8; i++) {
  const frequency = Math.round(between(400, 1000)), expected = `CQ DE W${i}ABC UR RST 599 TEST 73`, wpm = Math.round(between(16, 25))
  const audio = signal(expected, frequency, wpm, between(18, 25)), onset = between(2, 4), start = Math.round(onset * 8000), otherFrequency = frequency + Math.round(between(160, 250))
  const other = signal('CQ CQ DE K9XYZ UR 579 K CQ CQ DE K9XYZ UR 579 K', otherFrequency, Math.round(between(20, 32)), 35)
  for (let at = start; at < audio.length; at++) audio[at] += 1.3 * (other[at - start] ?? 0)
  await save('crowded', audio, expected, frequency, { otherFrequency, interfererOnset: onset })
}
for (let i = 0; i < 4; i++) {
  const oldFrequency = Math.round(between(400, 550)), frequency = Math.round(between(850, 1150))
  const first = signal('CQ 73', oldFrequency, 20, 30), second = signal('TEST 599 K', frequency, 20, 30)
  const gap = Math.round(between(3, 6) * 8000), audio = new Float32Array(first.length + gap + second.length)
  audio.set(first); audio.set(second, first.length + gap)
  await save('new-carrier', audio, 'CQ 73 TEST 599 K', frequency, { oldFrequency, onset: (first.length + gap) / 8000, liveOnly: true })
}
for (let i = 0; i < noiseCases; i++) {
  const amplitude = 10 ** between(-5, -.5)
  const audio = Float32Array.from({ length: 160000 }, () => amplitude * Math.sqrt(-2 * Math.log(Math.max(1e-12, rng()))) * Math.cos(2 * Math.PI * rng()))
  await save('noise', audio, '', 650, { amplitude })
}
const files = ['scripts/acquisition-benchmark.ts', 'src/signals.ts', 'src/morse.ts']
const sources = await Promise.all(files.map(async file => ({ file, sha256: sha256(await readFile(file)) })))
const manifest = JSON.stringify({ version: 1, seed, format: 'float32-le-mono', sampleRate: 8000, knownCarrier: false, sources, cases }, null, 2) + '\n'
await writeFile(join(directory, 'manifest.json'), manifest)
const traces = []
const originalFetch = globalThis.fetch
try {
  for (const item of cases) {
    const bytes = await readFile(join(directory, item.file))
    const audio = Float32Array.from({ length: bytes.length / 4 }, (_, i) => bytes.readFloatLE(i * 4))
    const scanner = new CWDecoder(8000, { frequency: 650, autoTune: true, bandwidth: 150 }), stream = new NeuralStream('cwformer')
    const chunks: Array<{ stop: number; frequency: number }> = [], history: Array<{ time: number; frequency: number }> = []
    let received = 0, blocks = 0
    globalThis.fetch = (async (input, init) => {
      const url = new URL(String(input), 'http://localhost')
      if (url.pathname.endsWith('/start')) return Response.json({ session: 'probe' })
      if (url.pathname.endsWith('/feed')) {
        const body = new Float32Array(init?.body as ArrayBuffer)
        if (!body.every((value, i) => value === audio[received + i])) throw new Error('Streaming reordered or changed audio')
        received += body.length
        chunks.push({ stop: received, frequency: Number(url.searchParams.get('frequency')) })
      }
      return Response.json({ text: '' })
    }) as typeof fetch
    await stream.start({ ...DEFAULT_RECEIVER, bandwidth: 150 })
    for (let at = 0; at < audio.length; at += 160) {
      const block = audio.subarray(at, at + 160)
      // Match the worklet: capture precedes decoding, readings arrive every third 20 ms block.
      stream.feed(block)
      const reading = scanner.process(block)
      if (block.length === 160 && ++blocks % 3 === 0) {
        stream.frequency = reading.frequency
        if (!history.length || history.at(-1)!.frequency !== reading.frequency) history.push({ time: (at + block.length) / 8000, frequency: reading.frequency })
      }
      await Promise.resolve(); await Promise.resolve(); await Promise.resolve()
    }
    const final = scanner.finish()
    stream.frequency = final.frequency
    await stream.finish()
    if (received !== audio.length) throw new Error(`${item.id}: lost audio ${received}/${audio.length}`)
    const fileScanner = new CWDecoder(8000, { frequency: 650, autoTune: true, bandwidth: 150 })
    for (let at = 0; at < audio.length; at += 256) fileScanner.process(audio.subarray(at, at + 256))
    traces.push({ id: item.id, chunks, history, finalFrequency: final.frequency, fileFrequency: fileScanner.finish().frequency })
  }
} finally { globalThis.fetch = originalFetch }
const pipelineFiles = [scannerPath, 'src/timing.ts', 'src/neural.ts', 'src/receiver.worklet.ts', 'src/file.worker.ts']
const pipeline = await Promise.all(pipelineFiles.map(async file => ({ file, sha256: sha256(await readFile(file)) })))
const output = join(directory, `${values.name}-traces.json`)
await writeFile(output, JSON.stringify({ version: 1, manifest: 'manifest.json', manifestSha256: sha256(manifest), protocol: 'Actual NeuralStream queue, zero transport latency, 20 ms samples then scanner readings every 60 ms; file uses a separate scanner with the file worker 256-sample blocks. No browser resampling or HTTP transport.', pipeline, traces }, null, 2) + '\n')
console.log(JSON.stringify({ manifest: join(directory, 'manifest.json'), traces: output, seed, cases: cases.length }))
