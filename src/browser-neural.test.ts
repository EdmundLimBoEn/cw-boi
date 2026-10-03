import { afterEach, beforeEach, expect, test } from 'bun:test'
import { DEFAULT_RECEIVER } from './decoder'
import { NeuralStream, cancelNeural, loadNeural, neuralDecode, subscribeModel, type ModelState } from './browser-neural'
import type { WorkerRequest, WorkerResponse } from './cwformer/protocol'

class FakeWorker {
  static instances: FakeWorker[] = []
  static hold = new Set<WorkerRequest['type']>()
  onmessage: ((event: MessageEvent<WorkerResponse>) => void) | null = null
  onerror: ((event: ErrorEvent) => void) | null = null
  requests: WorkerRequest[] = []
  terminated = false
  constructor() { FakeWorker.instances.push(this) }
  postMessage(request: WorkerRequest) {
    this.requests.push(request)
    if (!FakeWorker.hold.has(request.type)) queueMicrotask(() => this.reply({ id: request.id, type: 'result', text: request.type === 'finish' ? ' 73' : request.type === 'feed' ? this.requests.filter(item => item.type === 'feed').length === 1 ? 'CQ' : ' TEST' : '' }))
  }
  reply(data: WorkerResponse) { this.onmessage?.({ data } as MessageEvent<WorkerResponse>) }
  terminate() { this.terminated = true }
}

const originalWorker = globalThis.Worker
const originalFetch = globalThis.fetch
const settle = () => new Promise(resolve => setTimeout(resolve, 0))

beforeEach(() => {
  FakeWorker.instances = []; FakeWorker.hold.clear()
  globalThis.Worker = FakeWorker as unknown as typeof Worker
  globalThis.fetch = (() => { throw new Error('The browser client must never upload audio.') }) as typeof fetch
})
afterEach(() => {
  cancelNeural()
  globalThis.Worker = originalWorker; globalThis.fetch = originalFetch
})

test('model is lazy, reports progress, and a cancelled download can be retried', async () => {
  const states: ModelState[] = []
  const unsubscribe = subscribeModel(state => states.push(state))
  try {
    expect(FakeWorker.instances).toHaveLength(0)
    FakeWorker.hold.add('load')
    const first = loadNeural().then(() => 'loaded', error => error.name)
    const previous = FakeWorker.instances[0]
    const request = previous.requests[0]
    previous.reply({ id: request.id, type: 'progress', stage: 'download', loaded: 32, total: 100 })
    expect(states.at(-1)).toEqual({ phase: 'loading', stage: 'download', loaded: 32, total: 100 })
    cancelNeural()
    expect(await first).toBe('AbortError')
    expect(previous.terminated).toBe(true)
    FakeWorker.hold.clear()
    await loadNeural()
    previous.reply({ id: request.id, type: 'error', message: 'stale failure' })
    expect(states.at(-1)?.phase).toBe('ready')
    await loadNeural()
    expect(FakeWorker.instances).toHaveLength(2)
  } finally { unsubscribe() }
})

test('load failure releases the worker and retry creates a fresh one', async () => {
  const states: ModelState[] = []
  const unsubscribe = subscribeModel(state => states.push(state))
  try {
    FakeWorker.hold.add('load')
    const failed = loadNeural().catch(error => error.message)
    const worker = FakeWorker.instances[0]
    worker.reply({ id: worker.requests[0].id, type: 'error', message: 'Offline' })
    expect(await failed).toBe('Offline')
    expect(worker.terminated).toBe(true)
    expect(states.at(-1)).toEqual({ phase: 'error', message: 'Offline' })
    FakeWorker.hold.clear()
    await loadNeural()
    expect(FakeWorker.instances).toHaveLength(2)
  } finally { unsubscribe() }
})

test('finish drains queued audio once and retains a loaded model for the next stream', async () => {
  const stream = new NeuralStream()
  let text = ''
  stream.onText = value => { text = value }
  await stream.start(DEFAULT_RECEIVER)
  stream.feed(new Float32Array(8000)); stream.feed(new Float32Array(731))
  await Promise.all([stream.finish(), stream.finish()])
  const worker = FakeWorker.instances[0]
  expect(worker.requests.filter(request => request.type === 'feed').reduce((sum, request) => sum + request.samples!.length, 0)).toBe(8731)
  expect(worker.requests.filter(request => request.type === 'finish')).toHaveLength(1)
  expect(text).toBe('CQ TEST 73')
  const next = new NeuralStream()
  await next.start(DEFAULT_RECEIVER)
  expect(FakeWorker.instances).toHaveLength(1)
  expect(worker.requests.filter(request => request.type === 'load')).toHaveLength(1)
  next.cancel()
})

test('cancelling pending initialization cannot start capture or publish stale copy', async () => {
  FakeWorker.hold.add('start')
  const stream = new NeuralStream()
  let text = ''
  stream.onText = value => { text = value }
  const started = stream.start(DEFAULT_RECEIVER).then(() => 'started', error => error.name)
  await settle()
  const worker = FakeWorker.instances[0]
  const start = worker.requests.find(request => request.type === 'start')!
  stream.cancel()
  worker.reply({ id: start.id, type: 'result', text: 'STALE' })
  expect(await started).toBe('AbortError')
  stream.feed(new Float32Array(8000))
  await stream.finish()
  expect(worker.requests.some(request => request.type === 'feed')).toBe(false)
  expect(text).toBe('')
})

test('live backlog is bounded including in-flight audio and cancelled inference stays cancelled', async () => {
  FakeWorker.hold.add('feed')
  const stream = new NeuralStream()
  let text = '', error = ''
  stream.onText = value => { text = value }
  stream.onError = value => { error = (value as Error).message }
  await stream.start(DEFAULT_RECEIVER)
  for (let index = 0; index < 13; index++) stream.feed(new Float32Array(8000))
  const worker = FakeWorker.instances[0]
  expect(worker.terminated).toBe(true)
  expect(error).toContain('cannot decode live')
  const feed = worker.requests.find(request => request.type === 'feed')!
  worker.reply({ id: feed.id, type: 'result', text: 'STALE' })
  await stream.finish()
  expect(text).toBe('')
})

test('file decoding reports progress, preserves input, and abort releases inference', async () => {
  FakeWorker.hold.add('feed')
  const controller = new AbortController()
  const samples = Float32Array.of(0.25, 0.5, 0.75)
  let progress = 0
  const decoded = neuralDecode(samples, DEFAULT_RECEIVER, 'cwformer', controller.signal, value => { progress = value }).then(() => 'done', error => error.name)
  await settle()
  const worker = FakeWorker.instances[0]
  const feed = worker.requests.find(request => request.type === 'feed')!
  expect(feed.samples).not.toBe(samples)
  worker.reply({ id: feed.id, type: 'progress', stage: 'decode', loaded: 1, total: 2 })
  expect(progress).toBe(0.5)
  controller.abort()
  expect(await decoded).toBe('AbortError')
  expect(worker.terminated).toBe(true)
  expect(samples).toEqual(Float32Array.of(0.25, 0.5, 0.75))
})
