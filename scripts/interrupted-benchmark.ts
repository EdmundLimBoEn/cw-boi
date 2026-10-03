import { createHash } from 'node:crypto'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { parseArgs } from 'node:util'
import { makeSignal, randomSource, type Challenge } from '../src/signals'

const { values } = parseArgs({ options: {
  seed: { type: 'string' }, output: { type: 'string' },
  split: { type: 'string', default: 'dev' },
  'cases-per-condition': { type: 'string', default: '6' },
}, strict: true })
const seed = Number(values.seed), count = Number(values['cases-per-condition'])
if (!values.seed || !/^\d+$/.test(values.seed) || !Number.isInteger(seed) || seed < 1 || seed > 0xffffffff) throw new Error('--seed must be an integer in [1, 4294967295]')
if (!Number.isInteger(count) || count < 1 || count > 1000) throw new Error('--cases-per-condition must be an integer in [1, 1000]')
if (!['dev', 'final'].includes(values.split!)) throw new Error('--split must be dev or final')
const output = resolve(values.output ?? `.research/interrupted-${values.split}`)
const sampleRate = 8000, random = randomSource(seed)
const choose = <T>(items: readonly T[]): T => items[Math.floor(random() * items.length)]
const between = (low: number, high: number) => low + random() * (high - low)
const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
const group = () => Array.from({ length: 4 }, () => choose([...alphabet])).join('')
const call = () => `${choose(['W', 'K', 'N', 'G', 'F', 'DL', 'JA', 'VK', '9V'])}${choose([...'0123456789'])}${group().slice(0, 3)}`
const words = ['NORTH', 'RADIO', 'WATER', 'NIGHT', 'RAIN', 'GREEN', 'ROAD', 'EARLY', 'RIVER', 'WIND']
const message = (index: number) => index % 3 === 0 ? [`CQ DE ${call()}`, `UR RST ${choose(['339', '459', '579', '599'])}`, `73 DE ${call()} K`]
  : index % 3 === 1 ? [0, 1, 2].map(() => `${group()} ${group()}`)
    : [0, 1, 2].map(() => `${choose(words)} ${choose(words)}`)
const sha256 = (data: Uint8Array | string) => createHash('sha256').update(data).digest('hex')
const cases: Record<string, unknown>[] = []
await mkdir(output, { recursive: true })

for (const condition of ['clean-paused', 'weak-paused', 'crowded-paused']) {
  for (let index = 0; index < count; index++) {
    const texts = message(index), frequency = Math.round(between(450, 950)), wpm = Math.round(between(12, 34))
    const clean = condition === 'clean-paused', crowded = condition === 'crowded-paused'
    const snr = clean ? 30 : between(-12, -4), noiseRms = .4 / Math.sqrt(2) / 10 ** (snr / 20)
    const challenge: Challenge = { id: condition, name: condition, description: condition,
      frequency, wpm, snr: 100, jitter: clean ? 0 : .2, gapJitter: clean ? 0 : .25,
      drift: clean ? 0 : between(0, 12), fading: clean ? 0 : between(.45, .85),
      dashRatio: clean ? 3 : between(2.5, 3.8), swing: clean ? 0 : .15 }
    const pieces: { audio: Float32Array; startSample: number }[] = []
    const bursts: { expected: string; seed: number; startSample: number; lastMarkSample: number }[] = []
    let end = Math.round(between(1, 3) * sampleRate)
    for (const expected of texts) {
      const burstSeed = Math.floor(between(1, 0x7fffffff))
      const audio = makeSignal(expected, challenge, burstSeed, sampleRate)
      pieces.push({ audio, startSample: end })
      bursts.push({ expected, seed: burstSeed, startSample: end, lastMarkSample: end + audio.length - sampleRate })
      end += audio.length + choose([3, 7, 13, 24]) * sampleRate
    }
    const audio = new Float32Array(end), noiseSeed = Math.floor(between(1, 0x7fffffff)), noiseRandom = randomSource(noiseSeed)
    let crash = 0, crashState = 0
    for (let i = 0; i < audio.length; i++) {
      const gaussian = Math.sqrt(-2 * Math.log(Math.max(1e-12, noiseRandom()))) * Math.cos(2 * Math.PI * noiseRandom())
      if (crowded && noiseRandom() < 1 / sampleRate) crash = 1 + noiseRandom() * 2
      crash *= .998
      crashState = crashState * .85 + (noiseRandom() * 2 - 1) * .15
      audio[i] = gaussian * noiseRms + crash * crashState * 4
    }
    const interferenceHz = crowded ? choose([-1, 1]) * between(110, 220) : 0
    if (crowded) {
      const other = makeSignal(`${group()} ${group()} ${group()} ${group()}`, { ...challenge,
        frequency: frequency + interferenceHz, wpm: 31, jitter: .08, gapJitter: .08,
        drift: 0, fading: 0, dashRatio: 3, swing: 0 }, noiseSeed + 1, sampleRate)
      for (let i = 0; i < audio.length; i++) audio[i] += other[i % other.length] * .875
    }
    for (const piece of pieces) for (let i = 0; i < piece.audio.length; i++) audio[piece.startSample + i] += piece.audio[i]
    if (!audio.every(Number.isFinite)) throw new Error('Nonfinite audio')
    const id = `${condition}-${index + 1}`, file = `${id}.f32`, bytes = Buffer.alloc(audio.length * 4)
    for (let i = 0; i < audio.length; i++) bytes.writeFloatLE(audio[i], i * 4)
    await writeFile(join(output, file), bytes)
    cases.push({ id, condition, expected: texts.join(' '), file, sha256: sha256(bytes), sampleRate,
      samples: audio.length, durationSeconds: audio.length / sampleRate, frequency, bandwidth: 100,
      parameters: { ...challenge, snr, noiseSeed, interferenceHz, bursts,
        pauseScoringGraceSeconds: 2, trailingReceiverNoise: true } })
  }
}
const sources = await Promise.all(['src/signals.ts', 'src/morse.ts', 'scripts/interrupted-benchmark.ts'].map(async file => ({ file, sha256: sha256(await readFile(file)) })))
const manifest = { version: 1, split: `round3-interrupted-${values.split}`, seed, sampleRate,
  format: 'float32-le-mono', knownCarrier: true,
  snrDefinition: 'Keyed carrier RMS / broadband Gaussian noise RMS at 8 kHz, before fading. Receiver noise and neighboring station continue through all pauses.',
  provenance: { kind: 'synthetic', sources, text: 'Three complete independent bursts, then a long trailing pause. No waveform cropping or training generator. Burst last-mark times are accurate to one sample.' }, cases }
await writeFile(join(output, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n')
console.log(JSON.stringify({ manifest: join(output, 'manifest.json'), cases: cases.length,
  characters: cases.reduce((sum, item) => sum + String(item.expected).length, 0),
  durationSeconds: cases.reduce((sum, item) => sum + Number(item.durationSeconds), 0) }))
