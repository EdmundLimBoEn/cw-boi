import { expect, mock, test } from 'bun:test'
import ts from 'typescript'
import { CHALLENGES } from './signals'

// Execute the actual handler with audio and React state boundaries stubbed.
// This avoids a browser/model dependency while checking the array sent to playback.
const source = await Bun.file(new URL('./App.tsx', import.meta.url)).text()
const ast = ts.createSourceFile('App.tsx', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
let handler = ''
function visit(node: ts.Node) {
  if (ts.isFunctionDeclaration(node) && node.name?.text === 'playDemo') handler = node.getText(ast)
  ts.forEachChild(node, visit)
}
visit(ast)
if (!handler) throw new Error('Missing playDemo handler')
handler = ts.transpile(handler, { target: ts.ScriptTarget.ES2022 })

function setup() {
  const sample = CHALLENGES[0]
  const sampleAudio = new Float32Array([0.1, 0.2, 0.3])
  const generatedAudio = new Float32Array([0.4, 0.5])
  const makeSignal = mock(() => generatedAudio)
  const playSamples = mock(async () => true)
  const setError = mock(() => {})
  const noop = () => {}
  const scope = {
    sample, sampleAudio, makeSignal,
    LAB_TEXT: 'test message', SAMPLE_SEED: 947, SAMPLE_RATE: 8000,
    stop: noop, setError, setBusy: noop, setReading: noop, EMPTY: {},
    setNeuralText: noop, setReceiverTab: noop, setSampleId: noop,
    setSampleElapsed: noop, rx: {}, changeRx: noop, operation: { current: 0 },
    getAudio: () => ({ ready: async () => {}, loopback: async () => true, playSamples, context: { currentTime: 0 } }),
    startNeural: async () => {}, setSampleDuration: noop, setSourceName: noop,
    tx: { volume: 0.5 }, finishReception: noop, setSource: noop,
    startTime: { current: 0 }, audioError: String,
  }
  const playDemo = new Function(...Object.keys(scope), `${handler}; return playDemo`)(...Object.values(scope))
  return { playDemo, sample, sampleAudio, generatedAudio, makeSignal, playSamples, setError }
}

for (const explicit of [false, true]) {
  test(`selected sample reuses memoized audio on repeated playback (${explicit ? 'explicit' : 'default'} challenge)`, async () => {
    const { playDemo, sample, sampleAudio, makeSignal, playSamples, setError } = setup()
    for (let i = 0; i < 2; i++) await playDemo(...(explicit ? [sample] : []))
    expect(makeSignal).not.toHaveBeenCalled()
    expect(playSamples).toHaveBeenCalledTimes(2)
    for (const call of playSamples.mock.calls) {
      expect(call[0]).toBe(sampleAudio)
      expect(call[1]).toBe(8000)
      expect(call[2]).toBe(0.5)
    }
    expect(setError.mock.calls).toEqual([[''], ['']])
  })
}

test('a different explicit challenge synthesizes and plays its own audio', async () => {
  const { playDemo, generatedAudio, makeSignal, playSamples, setError } = setup()
  await playDemo(CHALLENGES[1])
  expect(makeSignal).toHaveBeenCalledTimes(1)
  expect(makeSignal).toHaveBeenCalledWith('test message', CHALLENGES[1], 947, 8000)
  expect(playSamples).toHaveBeenCalledTimes(1)
  expect(playSamples.mock.calls[0][0]).toBe(generatedAudio)
  expect(setError.mock.calls).toEqual([['']])
})
