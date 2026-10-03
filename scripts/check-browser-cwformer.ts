import * as ort from 'onnxruntime-web/wasm'
import { Frontend, MelFrontend } from '../src/cwformer/frontend'
import { Model, StreamingDecoder } from '../src/cwformer/runtime'

type DecodeReference = { fragments: string[]; tail: string; text: string }
type Case = {
  id: string; file: string; samples: number; frequency: number; audioSHA256: string
  reference: DecodeReference
  chunking?: Record<string, DecodeReference>
  retune?: { sample: number; frequency: number; chunkSize: number; reference: DecodeReference }
}
type Reference = {
  modelSHA256: string; pythonRuntimeSHA256: string; cases: Case[]
  frontend: {
    inputFile: string; blockSamples: number; warmupSamples: number
    cases: { bandwidth: number; blocks: { frequency: number; audio: string; mel: string; frames: number }[] }[]
  }
}

const query = new URLSearchParams(location.search)
const fixtures = query.get('fixtures') || '/.research/browser-reference/'
const modelRoot = query.get('modelRoot') || '/models/cwformer-v6/'
const wasmRoot = query.get('wasmRoot') || '/ort/'
const results: Record<string, unknown>[] = []

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message)
}

async function bytes(path: string) {
  const response = await fetch(path)
  assert(response.ok, `Fetch failed: ${path}, HTTP ${response.status}`)
  return new Uint8Array(await response.arrayBuffer())
}

async function floats(path: string) {
  const data = await bytes(path)
  return new Float32Array(data.buffer, data.byteOffset, data.length / 4)
}

async function digest(data: Uint8Array) {
  return [...new Uint8Array(await crypto.subtle.digest('SHA-256', data))].map(value => value.toString(16).padStart(2, '0')).join('')
}

function close(actual: Float32Array, expected: Float32Array, tolerance: number, label: string) {
  assert(actual.length === expected.length, `${label}: lengths ${actual.length} != ${expected.length}`)
  let maximum = 0, squared = 0
  for (let i = 0; i < actual.length; i++) {
    const error = Math.abs(actual[i] - expected[i])
    assert(Number.isFinite(error), `${label}: nonfinite value at ${i}`)
    maximum = Math.max(maximum, error)
    squared += error * error
  }
  assert(maximum <= tolerance, `${label}: maximum error ${maximum} > ${tolerance}`)
  results.push({ check: label, samples: actual.length, maxAbsoluteError: maximum, rmsError: Math.sqrt(squared / actual.length), tolerance })
}

async function compare(model: Model, item: Case, audio: Float32Array, size: number, reference: DecodeReference, retune = false) {
  const started = performance.now()
  const decoder = await StreamingDecoder.create(model, { frequency: item.frequency, bandwidth: 150 })
  const fragments = []
  try {
    for (let start = 0; start < audio.length; start += size) {
      if (retune && item.retune && start >= item.retune.sample) decoder.settings.frequency = item.retune.frequency
      fragments.push(await decoder.feed(audio.subarray(start, start + size)))
    }
    const tail = await decoder.flush()
    const text = fragments.join('') + tail
    const label = `${item.id}/${size}${retune ? '/retune' : ''}`
    assert(text === reference.text, `${label}: ${JSON.stringify(text)} != Python ${JSON.stringify(reference.text)}`)
    assert(JSON.stringify(fragments) === JSON.stringify(reference.fragments), `${label}: emission boundaries changed`)
    assert(tail === reference.tail, `${label}: flush changed`)
    assert(await decoder.flush() === '', `${label}: second flush emitted text`)
    let rejected = false
    try { await decoder.feed(audio.subarray(0, 1)) } catch { rejected = true }
    assert(rejected, `${label}: feed after flush accepted`)
    results.push({ check: label, text, chunks: fragments.length, exactPythonEmissionParity: true, elapsedMilliseconds: performance.now() - started })
    console.log(`PASS ${label}: ${JSON.stringify(text)}`)
  } finally { decoder.dispose() }
}

async function run() {
  const started = performance.now()
  ort.env.wasm.numThreads = 1
  ort.env.wasm.proxy = false
  ort.env.wasm.wasmPaths = { wasm: new URL(`${wasmRoot}ort-wasm-simd-threaded.wasm`, location.origin).href }
  const reference = await (await fetch(`${fixtures}reference.json`)).json() as Reference
  const manifest = await (await fetch(`${modelRoot}manifest.json`)).json() as { modelBytes: number; parts: { file: string; bytes: number; sha256: string }[] }
  const modelBytes = new Uint8Array(manifest.modelBytes)
  let offset = 0
  for (const part of manifest.parts) {
    const chunk = await bytes(`${modelRoot}${part.file}`)
    assert(chunk.length === part.bytes && await digest(chunk) === part.sha256, `Model part changed: ${part.file}`)
    modelBytes.set(chunk, offset)
    offset += chunk.length
  }
  assert(offset === modelBytes.length, 'Model size differs from manifest')
  const [window, basis] = await Promise.all([
    floats(`${modelRoot}mel_window.f32`),
    floats(`${modelRoot}mel_basis.f32`),
  ])
  assert(await digest(modelBytes) === reference.modelSHA256, 'Browser model differs from Python model')
  const input = await floats(`${fixtures}${reference.frontend.inputFile}`)
  for (const item of reference.frontend.cases) {
    const frontend = new Frontend(650, item.bandwidth)
    const mel = new MelFrontend(window, basis)
    mel.process(new Float32Array(reference.frontend.warmupSamples))
    for (let i = 0; i < item.blocks.length; i++) {
      const block = item.blocks[i]
      frontend.frequency = block.frequency
      const output = frontend.process(input.subarray(i * reference.frontend.blockSamples, (i + 1) * reference.frontend.blockSamples))
      close(output, await floats(`${fixtures}${block.audio}`), 2e-6, `frontend/${item.bandwidth}/${i}`)
      const features = mel.process(output)
      assert(features.frames === block.frames, `mel/${item.bandwidth}/${i}: frame count changed`)
      close(features.data, await floats(`${fixtures}${block.mel}`), 3e-5, `mel/${item.bandwidth}/${i}`)
    }
  }
  const model = await Model.create(modelBytes, window, basis)
  try {
    for (const item of reference.cases) {
      const audioBytes = await bytes(`${fixtures}${item.file}`)
      assert(await digest(audioBytes) === item.audioSHA256, `${item.id}: waveform checksum mismatch`)
      const audio = new Float32Array(audioBytes.buffer, audioBytes.byteOffset, audioBytes.length / 4)
      assert(audio.length === item.samples, `${item.id}: waveform length mismatch`)
      await compare(model, item, audio, 4000, item.reference)
      if (item.chunking) for (const [size, expected] of Object.entries(item.chunking)) await compare(model, item, audio, Number(size), expected)
      if (item.retune) await compare(model, item, audio, item.retune.chunkSize, item.retune.reference, true)
    }
    const invalid = await StreamingDecoder.create(model, { frequency: 650, bandwidth: 150 })
    try {
      for (const value of [NaN, Infinity, -Infinity]) {
        let rejected = false
        try { await invalid.feed(new Float32Array([value])) } catch { rejected = true }
        assert(rejected, `Invalid sample ${value} accepted`)
      }
      results.push({ check: 'finite-input-validation', passed: true })
    } finally { invalid.dispose() }
  } finally { await model.session.release() }
  return { passed: true, modelSHA256: reference.modelSHA256, pythonRuntimeSHA256: reference.pythonRuntimeSHA256,
    browser: navigator.userAgent, wasmThreads: 1, elapsedMilliseconds: performance.now() - started, results }
}

const state = window as unknown as { parityResult?: unknown; parityFinished?: boolean }
run().then(result => { state.parityResult = result }).catch(error => {
  state.parityResult = { passed: false, error: error instanceof Error ? error.stack : String(error), results }
}).finally(() => {
  state.parityFinished = true
  document.body.textContent = JSON.stringify(state.parityResult, null, 2)
})
