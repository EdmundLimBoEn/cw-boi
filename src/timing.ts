import { MORSE, REVERSE } from './morse'
import type { Run } from './decoder'

const median = (values: number[]) => values.length ? [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)] : 0
const prefixes = new Set(Object.values(MORSE).flatMap(code => Array.from({ length: code.length }, (_, i) => code.slice(0, i + 1))))
type Timing = { dot: number; dash: number; element: number; letter: number; word: number }
type Hypothesis = { text: string; pending: string; score: number }
const distance = (duration: number, mean: number) => Math.log(Math.max(0.002, duration) / mean) ** 2

function clusters(values: number[], initial: number[], rounds = 5) {
  const centers = [...initial]
  for (let iteration = 0; iteration < rounds; iteration++) {
    const groups: number[][] = centers.map(() => [])
    for (const value of values) {
      let nearest = 0
      for (let j = 1; j < centers.length; j++) if (distance(value, centers[j]) < distance(value, centers[nearest])) nearest = j
      groups[nearest].push(value)
    }
    groups.forEach((group, i) => { if (group.length >= 2) centers[i] = median(group) })
  }
  return centers
}

function learn(runs: Run[], initial: Timing): Timing {
  const marks = runs.filter(run => run.on && run.duration > initial.dot * 0.3 && run.duration < initial.dash * 2.5).map(run => run.duration)
  const gaps = runs.filter(run => !run.on && run.duration > initial.element * 0.25 && run.duration < initial.word * 2.5).map(run => run.duration)
  const [dot, dash] = clusters(marks, [initial.dot, initial.dash])
  const [element, letter, word] = clusters(gaps, [initial.element, initial.letter, initial.word])
  return {
    dot, dash: Math.max(dot * 1.8, dash), element,
    letter: Math.max(element * 1.7, letter), word: Math.max(letter * 1.55, word),
  }
}

function retain(hypotheses: Hypothesis[]) {
  const seen = new Set<string>()
  return hypotheses.sort((a, b) => a.score - b.score).filter(item => {
    const key = item.text + '|' + item.pending
    if (seen.has(key)) return false
    seen.add(key)
    return true
  }).slice(0, 12)
}

export function decodeTiming(runs: Run[], dit: number, gapUnit: number, adaptive: boolean, final = false) {
  const initial: Timing = { dot: dit, dash: dit * 3, element: dit, letter: gapUnit * 3, word: gapUnit * 7 }
  const global = adaptive ? learn(runs, initial) : initial
  let local = global
  let hypotheses: Hypothesis[] = [{ text: '', pending: '', score: 0 }]
  for (let index = 0; index < runs.length; index++) {
    // Local clusters follow a changing fist. No vocabulary model can rewrite a callsign.
    if (adaptive && index % 24 === 0) local = learn(runs.slice(Math.max(0, index - 48), index + 72), global)
    const run = runs[index]
    const next: Hypothesis[] = []
    const trailingGap = !final && index === runs.length - 1 && !run.on
    if (run.on) {
      if (run.duration < local.dot * 0.25) continue
      for (const hypothesis of hypotheses) {
        for (const symbol of ['.', '-']) {
          const code = hypothesis.pending + symbol
          const score = hypothesis.score + distance(run.duration, symbol === '.' ? local.dot : local.dash) / 0.065
          if (prefixes.has(code)) next.push({ ...hypothesis, pending: code, score })
        }
        if (!next.length) next.push({ text: hypothesis.text + '�', pending: '', score: hypothesis.score + 12 })
      }
    } else {
      for (const hypothesis of hypotheses) {
        const symbol = REVERSE[hypothesis.pending]
        const spacePenalty = trailingGap ? 0 : distance(run.duration, local.element) / 0.1
        if (hypothesis.pending && run.duration < (trailingGap ? Math.sqrt(local.element * local.letter) : local.letter * 1.35)) next.push({ ...hypothesis, score: hypothesis.score + spacePenalty })
        if (!hypothesis.pending) { next.push(hypothesis); continue }
        if (symbol && (!trailingGap || run.duration >= Math.sqrt(local.element * local.letter))) {
          const word = run.duration >= Math.sqrt(local.letter * local.word)
          for (const boundary of ['letter', 'word'] as const) {
            if (trailingGap && (boundary === 'word') !== word) continue
            next.push({ text: hypothesis.text + symbol + (boundary === 'word' ? ' ' : ''), pending: '', score: hypothesis.score + (trailingGap ? 0 : distance(run.duration, local[boundary]) / 0.12) })
          }
        }
        if (!symbol && run.duration >= Math.sqrt(local.element * local.letter)) next.push({ text: hypothesis.text + '�', pending: '', score: hypothesis.score + 8 })
      }
    }
    hypotheses = retain(next.length ? next : hypotheses)
  }
  if (final) hypotheses = retain(hypotheses.map(hypothesis => ({ ...hypothesis, text: hypothesis.text + (hypothesis.pending ? REVERSE[hypothesis.pending] ?? '�' : ''), pending: '' })))
  const best = hypotheses[0]
  const alternative = hypotheses.find(item => item.text.trim() !== best.text.trim())
  const margin = alternative ? alternative.score - best.score : Infinity
  const alternatives = hypotheses.filter(item => item.text.trim() !== best.text.trim() && item.score - best.score < 2).slice(0, 2).map(item => item.text.trim())
  return { text: best.text, pending: best.pending, confidence: best.text || best.pending ? Math.round(Math.min(100, margin * 25)) : 0, alternatives }
}
