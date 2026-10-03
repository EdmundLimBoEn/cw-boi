import * as ort from 'onnxruntime-web/wasm'
import { Model, StreamingDecoder } from './runtime'
import type { ModelProgress, WorkerRequest, WorkerResponse } from './protocol'

type Asset = { file: string; bytes: number; sha256: string }
type Manifest = {
  format: 'cwformer-v6'
  modelBytes: number
  modelSha256: string
  parts: Asset[]
  window: Asset
  basis: Asset
}

const MODEL_SHA256 = '8344b39dba22e595ea8170bd4ffdb514aaef1c4b67b7de3ef965b5f66a081374'
const WINDOW_SHA256 = 'c692d19f0b2bd745f0dfc967261c3e3d21a2473e485a677d219e9b4a73e0218e'
const BASIS_SHA256 = '3605837643860610e8505c4818f58f8644df705b3ce61e3f2148cbd78e93db5e'

const worker = self as unknown as {
  location: Location
  onmessage: (event: MessageEvent<WorkerRequest>) => void
  postMessage: (message: WorkerResponse) => void
}
const assetRoot = new URL('/models/cwformer-v6/', worker.location.origin)
// Keep ORT's bundled loader; importing a public .mjs at runtime breaks Vite development.
ort.env.wasm.wasmPaths = { wasm: new URL('/ort/ort-wasm-simd-threaded.wasm', worker.location.origin).href }

async function download(path: string, progress?: (bytes: number) => void, expectedBytes?: number) {
  const url = new URL(path, assetRoot)
  if (url.origin !== assetRoot.origin || !url.pathname.startsWith(assetRoot.pathname)) throw new Error('Invalid model asset URL.')
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Could not download the CWformer model (${response.status}). Please retry.`)
  if (expectedBytes !== undefined && response.body) {
    const bytes = new Uint8Array(expectedBytes)
    const reader = response.body.getReader()
    let at = 0
    let lastProgressAt = -Infinity
    try {
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        if (at + value.length > bytes.length) throw new Error('The model download has an invalid size.')
        bytes.set(value, at)
        at += value.length
        const now = performance.now()
        if (now - lastProgressAt >= 100 || at === bytes.length) {
          lastProgressAt = now
          progress?.(at)
        }
      }
      if (at !== bytes.length) throw new Error('The model download was incomplete. Please retry.')
      return bytes
    } finally { await reader.cancel() }
  }
  const bytes = new Uint8Array(await response.arrayBuffer())
  progress?.(bytes.length)
  return bytes
}

async function sha256(bytes: Uint8Array) {
  const digest = await crypto.subtle.digest('SHA-256', bytes.buffer as ArrayBuffer)
  return Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, '0')).join('')
}

async function loadModel(progress: (progress: ModelProgress) => void) {
  const manifest = JSON.parse(new TextDecoder().decode(await download('manifest.json'))) as Manifest
  if (manifest.format !== 'cwformer-v6' || manifest.modelBytes !== 83222226 || !Array.isArray(manifest.parts) || manifest.parts.length > 16 || manifest.modelSha256 !== MODEL_SHA256 || manifest.window?.bytes !== 1600 || manifest.window.sha256 !== WINDOW_SHA256 || manifest.basis?.bytes !== 32160 || manifest.basis.sha256 !== BASIS_SHA256) throw new Error('Invalid CWformer model manifest.')
  const bytes = new Uint8Array(manifest.modelBytes)
  let loaded = 0
  progress({ stage: 'download', loaded, total: bytes.length })
  for (const part of manifest.parts) {
    if (!Number.isInteger(part.bytes) || part.bytes <= 0 || loaded + part.bytes > bytes.length) throw new Error('Invalid CWformer model part size.')
    const data = await download(part.file, current => progress({ stage: 'download', loaded: loaded + current, total: bytes.length }), part.bytes)
    if (data.length !== part.bytes || await sha256(data) !== part.sha256) throw new Error('The model download was incomplete or damaged. Please reload and retry.')
    bytes.set(data, loaded)
    loaded += data.length
    progress({ stage: 'download', loaded, total: bytes.length })
  }
  if (loaded !== bytes.length || await sha256(bytes) !== manifest.modelSha256) throw new Error('The model download failed its integrity check. Please reload and retry.')
  const [window, basis] = await Promise.all([download(manifest.window.file), download(manifest.basis.file)])
  if (window.length !== manifest.window.bytes || basis.length !== manifest.basis.bytes || await sha256(window) !== WINDOW_SHA256 || await sha256(basis) !== BASIS_SHA256) throw new Error('The CWformer audio frontend failed its integrity check. Please reload and retry.')
  progress({ stage: 'initialize', loaded: 0, total: 0 })
  return Model.create(bytes, new Float32Array(window.buffer), new Float32Array(basis.buffer))
}

let model: Promise<Model> | undefined
let decoder: StreamingDecoder | undefined
let queue = Promise.resolve()

worker.onmessage = ({ data: request }) => {
  queue = queue.then(async () => {
    const progress = (value: ModelProgress) => worker.postMessage({ id: request.id, type: 'progress', ...value })
    try {
      if (!Number.isInteger(request.id) || !['load', 'start', 'feed', 'finish'].includes(request.type)) throw new Error('Invalid decoder request.')
      let text = ''
      if (request.type === 'load' || request.type === 'start') {
        model ??= loadModel(progress).catch(error => { model = undefined; throw error })
        const ready = await model
        if (request.type === 'start') {
          decoder?.dispose()
          decoder = undefined
          decoder = await StreamingDecoder.create(ready, request)
        }
      } else {
        if (!decoder) throw new Error('Start the decoder before sending audio.')
        decoder.settings = { frequency: request.frequency, bandwidth: request.bandwidth }
        if (request.type === 'feed') {
          if (!(request.samples instanceof Float32Array)) throw new Error('Audio samples are required.')
          let lastProgress = 0
          text = await decoder.feed(request.samples, loaded => {
            if (request.samples!.length >= 16000 && (loaded - lastProgress >= 8000 || loaded === request.samples!.length)) {
              lastProgress = loaded
              progress({ stage: 'decode', loaded, total: request.samples!.length })
            }
          })
        } else text = await decoder.flush()
      }
      worker.postMessage({ id: request.id, type: 'result', text })
    } catch (error) {
      decoder?.dispose()
      decoder = undefined
      worker.postMessage({ id: request.id, type: 'error', message: error instanceof Error ? error.message : String(error) })
    }
  })
}
