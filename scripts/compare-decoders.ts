import { CHALLENGES, characterErrors, makeSignal } from '../src/signals'
import { decodeSamples } from '../src/decoder'

const text = 'CQ DE 9V1ABC UR RST 579 QTH SINGAPORE K'
const endpoint = process.env.NEURAL_URL ?? 'http://127.0.0.1:8787'
const health = await fetch(`${endpoint}/api/health`)
if (!health.ok) throw new Error('Neural service is unavailable.')
const model = await health.json()
const results = []
for (const challenge of CHALLENGES) {
  const samples = makeSignal(text, challenge, 39181)
  const adaptive = decodeSamples(samples, 8000, { frequency: challenge.frequency, autoTune: false })
  const started = performance.now()
  const response = await fetch(`${endpoint}/api/decode?frequency=${challenge.frequency}&bandwidth=100`, { method: 'POST', body: samples.buffer })
  if (!response.ok) throw new Error(await response.text())
  const neural = await response.json() as { text: string }
  const result = { challenge: challenge.id, expected: text, adaptive: adaptive.text.trim(), neural: neural.text, adaptiveCER: characterErrors(text, adaptive.text).cer, neuralCER: characterErrors(text, neural.text).cer, neuralMs: Math.round(performance.now() - started) }
  results.push(result)
  if (!process.argv.includes('--json')) console.log(JSON.stringify(result))
}
if (process.argv.includes('--json')) console.log(JSON.stringify({ seed: 39181, knownCarrier: true, sampleRate: 8000, model, results }, null, 2))
