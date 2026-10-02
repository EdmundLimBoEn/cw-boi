import { createHash } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, relative, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { parseArgs } from 'node:util'
import { decodeSamples } from '../src/decoder'
import { characterErrors } from '../src/signals'

const { values } = parseArgs({ args: process.argv.slice(2), options: {
  manifest: { type: 'string' }, output: { type: 'string' }, baseline: { type: 'string' },
} })
if (!values.manifest) throw new Error('Usage: bun scripts/benchmark-adaptive.ts --manifest PATH [--baseline COMMIT] [--output REPORT]')
const sha256 = (data: string | Uint8Array) => createHash('sha256').update(data).digest('hex')
const manifestPath = resolve(values.manifest), root = dirname(manifestPath)
const manifestBytes = await readFile(manifestPath), manifest = JSON.parse(manifestBytes.toString())
if (manifest.format !== 'float32-le-mono' || manifest.sampleRate !== 8000 || !Array.isArray(manifest.cases)) throw new Error('Expected an 8 kHz float32-le-mono corpus manifest')
const files = ['src/decoder.ts', 'src/timing.ts', 'src/morse.ts']
const sources = await Promise.all(files.map(async file => ({ file, sha256: sha256(await readFile(file)) })))
type Result = { id: string; condition: string; expected: string; actual: string; errors: number; cer: number; errorsWithoutSpaces: number; durationSeconds: number; inferenceSeconds: number }
const summary = (rows: Result[]) => {
  const speech = rows.filter(row => row.expected), noise = rows.filter(row => !row.expected)
  const characters = speech.reduce((sum, row) => sum + row.expected.length, 0), errors = speech.reduce((sum, row) => sum + row.errors, 0)
  const noiseSeconds = noise.reduce((sum, row) => sum + row.durationSeconds, 0), falseCharacters = noise.reduce((sum, row) => sum + row.actual.replaceAll(' ', '').length, 0)
  return { cases: rows.length, speechCases: speech.length, characters, errors, cer: characters ? errors / characters : null,
    exact: speech.filter(row => row.errors === 0).length, noiseCases: noise.length, noiseSeconds, falseCharacters,
    falseCharactersPerMinute: noiseSeconds ? falseCharacters * 60 / noiseSeconds : null,
    audioSeconds: rows.reduce((sum, row) => sum + row.durationSeconds, 0), inferenceSeconds: rows.reduce((sum, row) => sum + row.inferenceSeconds, 0) }
}
const audio = await Promise.all(manifest.cases.map(async (item: { file: string; sha256: string; sampleRate: number; samples: number }) => {
  const path = resolve(root, item.file)
  if (relative(root, path).startsWith('..')) throw new Error('Audio path must remain inside the corpus directory')
  const bytes = await readFile(path)
  if (sha256(bytes) !== item.sha256 || bytes.length % 4 || item.sampleRate !== 8000 || bytes.length / 4 !== item.samples) throw new Error(`Invalid audio or checksum: ${item.file}`)
  const samples = new Float32Array(Uint8Array.from(bytes).buffer)
  if (!samples.every(Number.isFinite)) throw new Error(`Non-finite audio: ${item.file}`)
  return samples
}))
function evaluate(decoder: typeof decodeSamples) {
  const results: Result[] = manifest.cases.map((item: { id: string; condition: string; expected: string; frequency: number; bandwidth: number }, i: number) => {
    const expected = item.expected.trim().toUpperCase(), start = performance.now()
    const actual = decoder(audio[i], 8000, { frequency: item.frequency, bandwidth: item.bandwidth, autoTune: !manifest.knownCarrier }).text.trim().toUpperCase()
    return { id: item.id, condition: item.condition, expected, actual, ...characterErrors(expected, actual),
      errorsWithoutSpaces: characterErrors(expected.replaceAll(' ', ''), actual.replaceAll(' ', '')).errors,
      durationSeconds: audio[i].length / 8000, inferenceSeconds: (performance.now() - start) / 1000 }
  })
  return { results, summary: summary(results), conditions: Object.fromEntries([...new Set(results.map(row => row.condition))].sort().map(condition => [condition, summary(results.filter(row => row.condition === condition))])) }
}
let baseline
if (values.baseline) {
  if (!/^[0-9a-f]{7,40}$/i.test(values.baseline)) throw new Error('--baseline must be a Git commit hash')
  const commit = execFileSync('git', ['rev-parse', values.baseline], { encoding: 'utf8' }).trim()
  const directory = await mkdtemp(join(tmpdir(), 'cw-adaptive-baseline-'))
  try {
    const sources = await Promise.all(files.map(async file => {
      const content = execFileSync('git', ['show', `${commit}:${file}`])
      await writeFile(join(directory, file.slice(4)), content)
      return { file, sha256: sha256(content) }
    }))
    const { decodeSamples: original } = await import(pathToFileURL(join(directory, 'decoder.ts')).href)
    baseline = { commit, sources, ...evaluate(original) }
  } finally { await rm(directory, { recursive: true, force: true }) }
}
const report = { version: 1, split: manifest.split, seed: manifest.seed, manifest: relative(process.cwd(), manifestPath), manifestSHA256: sha256(manifestBytes),
  engine: 'adaptive', mode: 'stream', chunkSamples: 256, knownCarrier: manifest.knownCarrier, sources,
  benchmarkSHA256: sha256(await readFile(import.meta.filename)), complete: true, ...evaluate(decodeSamples), ...(baseline ? { baseline } : {}) }
if (values.output) await writeFile(values.output, JSON.stringify(report, null, 2) + '\n')
else console.log(JSON.stringify(report, null, 2))
console.error(JSON.stringify({ output: values.output, summary: report.summary, baseline: baseline?.summary }))
