import { expect, test } from 'bun:test'
import { inferDit, type Run } from './decoder'
import { transmission } from './morse'
import { characterErrors, randomSource } from './signals'
import { decodeTiming } from './timing'

test('separates uneven element gaps from letters while tempo and word pauses change', () => {
  let errors = 0, characters = 0
  for (let sample = 0; sample < 40; sample++) {
    const random = randomSource(8742913 + sample * 271)
    const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    const text = Array.from({ length: 6 }, () => Array.from({ length: 5 }, () => alphabet[Math.floor(random() * alphabet.length)]).join('')).join(' ')
    const wpm = [10, 18, 28, 40][sample % 4]
    const tones = transmission(text, wpm).tones
    const runs: Run[] = []
    let previousEnd = 0
    for (let i = 0; i < tones.length; i++) {
      const tone = tones[i]
      const tempo = (1 + 0.3 * Math.sin(i / 17)) * (i > tones.length / 2 ? 1.35 : 1)
      let gap = (tone.start - previousEnd) * tempo * (1 + 0.45 * (random() * 2 - 1))
      if (tone.start - previousEnd > 1.2 / wpm * 5 && random() < 0.45) gap *= 2 + random() * 3
      const duration = (tone.end - tone.start) * tempo * (1 + 0.35 * (random() * 2 - 1)) * (tone.symbol === '-' ? 4.1 / 3 : 1)
      if (i) runs.push({ on: false, duration: gap })
      runs.push({ on: true, duration })
      previousEnd = tone.end
    }
    const dit = inferDit(runs, 0.06)
    const copy = decodeTiming(runs, dit, dit, true, true).text
    errors += characterErrors(text, copy).errors
    characters += text.length
  }
  expect(errors / characters).toBeLessThan(0.075)
})
