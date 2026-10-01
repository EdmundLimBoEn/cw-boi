import { decodeSamples } from '../src/decoder'
import { CHALLENGES, characterErrors, makeSignal, randomSource, type Challenge } from '../src/signals'

const random = randomSource(92017)
const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
const text = () => Array.from({ length: 6 }, () => Array.from({ length: 5 }, () => alphabet[Math.floor(random() * alphabet.length)]).join('')).join(' ')
const cases: { name: string; challenge: Challenge; seed: number; text: string }[] = []
for (const wpm of [10, 18, 28, 40]) {
  for (const jitter of [0, 0.2, 0.35]) {
    for (const dashRatio of [2.6, 3.6, 4.4]) {
      cases.push({ name: `hand-${wpm}wpm-${jitter * 100}pct-${dashRatio}dah`, text: text(), seed: 71831 + cases.length * 97, challenge: { ...CHALLENGES[1], wpm, jitter, dashRatio, swing: 0.25, snr: 15 } })
    }
  }
}
for (const snr of [15, 5, 0, -5]) {
  for (const frequency of [400, 650, 1000]) cases.push({ name: `radio-${snr}db-${frequency}hz`, text: text(), seed: 12983 + cases.length * 31, challenge: { ...CHALLENGES[2], frequency, snr, fading: 0.5, jitter: 0.15 } })
}
const results = []
let edits = 0, characters = 0
for (const test of cases) {
  const start = performance.now()
  const signal = makeSignal(test.text, test.challenge, test.seed)
  const decoded = decodeSamples(signal, 8000)
  const score = characterErrors(test.text, decoded.text)
  edits += score.errors; characters += test.text.length
  results.push({ name: test.name, ...score, expected: test.text, actual: decoded.text.trim(), milliseconds: Math.round(performance.now() - start) })
}
const report = { suite: 'synthetic-v1', sampleRate: 8000, autoTune: true, seed: 92017, disclaimer: 'Synthetic regression tests only; no real-world accuracy or superiority claim.', total: cases.length, exact: results.filter(result => result.errors === 0).length, cer: edits / characters, results }
if (process.argv.includes('--json')) console.log(JSON.stringify(report, null, 2))
else {
  console.table(results.map(({ name, errors, cer, milliseconds }) => ({ case: name, errors, 'CER %': (cer * 100).toFixed(2), ms: milliseconds })))
  console.log(`${report.exact}/${report.total} exact; ${(report.cer * 100).toFixed(2)}% aggregate CER (${edits}/${characters}). Synthetic only.`)
}
