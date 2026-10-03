import { createHash } from 'node:crypto'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { resolve, join } from 'node:path'
import { parseArgs } from 'node:util'
import { CHALLENGES, makeSignal, randomSource } from '../src/signals'

const { values } = parseArgs({ options: {
  split: { type: 'string', default: 'dev' },
  output: { type: 'string' },
  seed: { type: 'string' },
  'cases-per-condition': { type: 'string', default: '3' },
  'noise-repeats': { type: 'string', default: '1' },
}, strict: true })
if (!['dev', 'final'].includes(values.split!)) throw new Error('--split must be dev or final')
const casesPerCondition = Number(values['cases-per-condition']), noiseRepeats = Number(values['noise-repeats'])
if (![casesPerCondition, noiseRepeats].every(value => Number.isInteger(value) && value >= 1 && value <= 1000)) throw new Error('Case counts must be integers in [1, 1000]')
const split = values.split!, seed = values.seed === undefined ? (split === 'dev' ? 817351 : 917351) : Number(values.seed)
if ((values.seed !== undefined && !/^\d+$/.test(values.seed)) || !Number.isInteger(seed) || seed < 1 || seed > 0xffffffff) throw new Error('--seed must be an integer in [1, 4294967295]')
const output = resolve(values.output ?? `.research/continuous-${split}`)
const sampleRate = 8000
const random = randomSource(seed)
const choose = <T>(items: readonly T[]): T => items[Math.floor(random() * items.length)]
const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', digits = '0123456789'
const pickLetters = (count: number) => Array.from({ length: count }, () => choose([...letters])).join('')
const call = () => `${choose(['W', 'K', 'N', 'G', 'M', 'F', 'DL', 'JA', 'VK', 'ZS', '9V'])}${choose([...digits])}${pickLetters(2 + Math.floor(random() * 2))}`
const words = ['RAIN', 'WIND', 'NORTH', 'SOUTH', 'CLOUD', 'LIGHT', 'RIVER', 'HILL', 'VALLEY', 'GREEN', 'CLEAR', 'RADIO', 'TODAY', 'AFTER', 'EARLY', 'NIGHT', 'WATER', 'STONE', 'BRIDGE', 'ROAD']
const message = (index: number) => index % 3 === 0
  ? `CQ DE ${call()} UR RST ${choose(['339', '459', '579', '599'])} NAME ${choose(['ALAN', 'FRED', 'JANE', 'MARY', 'PETER', 'JOHN'])} K`
  : index % 3 === 1
    ? Array.from({ length: 5 }, () => Array.from({ length: 4 }, () => choose([...(letters + digits)])).join('')).join(' ')
    : Array.from({ length: 7 }, () => choose(words)).join(' ')
const sha256 = (data: Uint8Array | string) => createHash('sha256').update(data).digest('hex')
const cases: Record<string, unknown>[] = []
await mkdir(output, { recursive: true })

async function save(id: string, condition: string, expected: string, audio: Float32Array, frequency: number, parameters: unknown) {
  if (!audio.every(Number.isFinite)) throw new Error(`${id}: nonfinite audio`)
  const file = `${id}.f32`
  const bytes = Buffer.alloc(audio.length * 4)
  for (let i = 0; i < audio.length; i++) bytes.writeFloatLE(audio[i], i * 4)
  await writeFile(join(output, file), bytes)
  cases.push({ id, condition, expected, file, sha256: sha256(bytes), sampleRate,
    samples: audio.length, durationSeconds: audio.length / sampleRate, frequency,
    bandwidth: 100, parameters })
}

for (const base of CHALLENGES) {
  for (let index = 0; index < casesPerCondition; index++) {
    const caseSeed = Math.floor(random() * 0x7fffffff) + 1
    const challenge = { ...base, wpm: Math.round(base.wpm * (0.85 + random() * 0.3)),
      frequency: Math.round(450 + random() * 500), snr: base.snr + (random() - 0.5) * 4 }
    let expected = message(index), body = makeSignal(expected, challenge, caseSeed)
    for (let attempt = 0; attempt < 100; attempt++) {
      if (body.length >= sampleRate * 12 && body.length <= sampleRate * 28) break
      expected = body.length > sampleRate * 28 ? expected.split(' ').slice(0, -1).join(' ') : `${expected} ${choose(words)}`
      body = makeSignal(expected, challenge, caseSeed)
    }
    if (body.length < sampleRate * 12 || body.length > sampleRate * 28) throw new Error('Could not make a complete 12–28 second message')
    const lead = Math.round((0.15 + random() * 1.5) * sampleRate)
    const audio = new Float32Array(lead + body.length)
    audio.set(body, lead)
    await save(`${base.id}-${index + 1}`, base.id, expected, audio, challenge.frequency,
      { seed: caseSeed, leadingSilenceSeconds: lead / sampleRate, ...challenge })
  }
}

for (const [index, kind] of ['silence', 'white-quiet', 'white-loud', 'colored', 'impulsive', 'steady-carrier'].entries()) {
  for (let repeat = 0; repeat < noiseRepeats; repeat++) {
    const caseSeed = Math.floor(random() * 0x7fffffff) + 1
    const noiseRandom = randomSource(caseSeed)
    const audio = new Float32Array(sampleRate * 20)
    const frequency = Math.round(450 + random() * 500)
    let colored = 0, crash = 0
    for (let i = 0; i < audio.length; i++) {
      const gaussian = Math.sqrt(-2 * Math.log(Math.max(1e-12, noiseRandom()))) * Math.cos(2 * Math.PI * noiseRandom())
      colored = 0.93 * colored + 0.07 * gaussian
      if (noiseRandom() < 1.5 / sampleRate) crash = 3
      crash *= 0.998
      audio[i] = kind === 'silence' ? 0 : kind === 'white-quiet' ? gaussian * 0.01
        : kind === 'white-loud' ? gaussian * 0.8 : kind === 'colored' ? colored
        : kind === 'impulsive' ? gaussian * (0.03 + crash)
        : gaussian * 0.03 + 0.4 * Math.sin(2 * Math.PI * frequency * i / sampleRate)
    }
    await save(`noise-${index + 1}${noiseRepeats > 1 ? `-${repeat + 1}` : ''}`, 'noise', '', audio, frequency, { seed: caseSeed, kind })
  }
}
const sources = await Promise.all(['src/signals.ts', 'src/morse.ts', 'scripts/continuous-benchmark.ts'].map(async file => ({ file, sha256: sha256(await readFile(file)) })))
const manifest = { version: 1, split, seed, sampleRate, format: 'float32-le-mono', knownCarrier: true,
  snrDefinition: 'Keyed carrier RMS / broadband Gaussian noise RMS at 8 kHz; before fading.',
  provenance: { kind: 'synthetic', sources, text: 'Random callsigns, reports, word sequences and alphanumeric groups. No Python training generator used.' }, cases }
await writeFile(join(output, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n')
console.log(JSON.stringify({ manifest: join(output, 'manifest.json'), split, cases: cases.length,
  speechCases: cases.filter(item => item.expected).length, durationSeconds: cases.reduce((sum, item) => sum + Number(item.durationSeconds), 0) }))
