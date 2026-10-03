import { expect, test } from 'bun:test'
import { DEFAULT_RECEIVER } from './decoder'
import { NeuralStream } from './neural'

test('finishing twice drains audio once and closes the remote session once', async () => {
  const original = globalThis.fetch
  const calls: string[] = []
  let received = 0
  globalThis.fetch = (async (input, init) => {
    const url = String(input); calls.push(url)
    await new Promise(resolve => setTimeout(resolve, 1))
    if (url.includes('/start')) return Response.json({ session: 'test-session' })
    if (url.includes('/feed')) received += (init?.body as ArrayBuffer).byteLength / 4
    return Response.json({ text: url.includes('/finish') ? 'CQ TEST' : 'CQ' })
  }) as typeof fetch
  try {
    const stream = new NeuralStream()
    let text = ''
    stream.onText = copy => { text = copy }
    await stream.start(DEFAULT_RECEIVER)
    stream.feed(new Float32Array(8000))
    stream.feed(new Float32Array(731))
    await Promise.all([stream.finish(), stream.finish()])
    expect(received).toBe(8731)
    expect(calls.filter(url => url.includes('/finish'))).toHaveLength(1)
    expect(text).toBe('CQ TEST')
  } finally { globalThis.fetch = original }
})

test('cancelling a pending start releases the late session', async () => {
  const original = globalThis.fetch
  const calls: string[] = []
  let resolveStart!: (value: Response) => void
  globalThis.fetch = ((input) => {
    const url = String(input); calls.push(url)
    return url.includes('/start') ? new Promise(resolve => { resolveStart = resolve }) : Promise.resolve(Response.json({ text: '' }))
  }) as typeof fetch
  try {
    const stream = new NeuralStream()
    const started = stream.start(DEFAULT_RECEIVER)
    stream.cancel()
    resolveStart(Response.json({ session: 'late-session' }))
    await started
    expect(calls.some(url => url.includes('/cancel?session=late-session'))).toBe(true)
  } finally { globalThis.fetch = original }
})

test('engine selection follows file and stream requests, and cancelled copy stays cancelled', async () => {
  const original = globalThis.fetch
  const calls: URL[] = []
  let resolveFeed!: (value: Response) => void
  globalThis.fetch = ((input) => {
    const url = new URL(String(input), 'http://localhost'); calls.push(url)
    if (url.pathname.endsWith('/start')) return Promise.resolve(Response.json({ session: 'cwformer-session' }))
    if (url.pathname.endsWith('/feed')) return new Promise(resolve => { resolveFeed = resolve })
    return Promise.resolve(Response.json({ text: 'CQ' }))
  }) as typeof fetch
  try {
    const { neuralDecode } = await import('./neural')
    expect(await neuralDecode(new Float32Array(160), DEFAULT_RECEIVER, 'cwformer')).toBe('CQ')
    const stream = new NeuralStream('cwformer')
    let text = ''
    stream.onText = copy => { text = copy }
    await stream.start(DEFAULT_RECEIVER)
    stream.feed(new Float32Array(8000))
    stream.cancel()
    resolveFeed(Response.json({ text: 'OLD COPY' }))
    await stream.finish()
    expect(text).toBe('')
    expect(calls.map(url => url.pathname)).toEqual(['/api/decode', '/api/stream/start', '/api/stream/feed', '/api/stream/cancel'])
    expect(calls.every(url => url.searchParams.get('engine') === 'cwformer')).toBe(true)
  } finally { globalThis.fetch = original }
})
