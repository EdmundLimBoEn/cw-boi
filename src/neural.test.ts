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
