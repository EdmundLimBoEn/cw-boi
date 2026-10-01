import { describe, expect, test } from 'bun:test'
import { CWDecoder, decodeSamples } from './decoder'
import { decodeText, encodeText, synthesize, transmission, wavBytes } from './morse'
import { CHALLENGES, characterErrors, makeSignal, randomSource } from './signals'

describe('Morse timing and audio', () => {
  test('round trips international Morse and joined prosigns', () => {
    expect(decodeText(encodeText('CQ DE 9V1ABC 599? <SK>'))).toBe('CQ DE 9V1ABC 599? <SK>')
    expect(decodeText('... --- ... / .- -...')).toBe('SOS AB')
    expect(decodeText('··· −−− ···   ·− −···')).toBe('SOS AB')
  })
  test('uses exact 1:3 marks, 1:3:7 gaps, and PARIS Farnsworth spacing', () => {
    const plan = transmission('ET E', 20)
    expect(plan.tones[0].end - plan.tones[0].start).toBeCloseTo(0.06)
    expect(plan.tones[1].end - plan.tones[1].start).toBeCloseTo(0.18)
    expect(plan.tones[1].start - plan.tones[0].end).toBeCloseTo(0.18)
    expect(plan.tones[2].start - plan.tones[1].end).toBeCloseTo(0.42)
    const farnsworth = transmission('PARIS PARIS', 25, 10)
    const nextP = farnsworth.tones.find(tone => tone.index === 6)!
    expect(nextP.start - farnsworth.tones[0].start).toBeCloseTo(6)
    expect(() => transmission('HELLO', 20, 25)).toThrow()
    expect(() => transmission('A'.repeat(1001), 20)).toThrow()
    expect(transmission('HELLO 😊', 20).unsupported).toEqual(['😊'])
  })
  test('exports valid mono PCM WAV', () => {
    const pcm = synthesize(transmission('CQ', 20), 650)
    const wav = wavBytes(pcm)
    const view = new DataView(wav)
    expect(new TextDecoder().decode(wav.slice(0, 4))).toBe('RIFF')
    expect(view.getUint32(24, true)).toBe(22050)
    expect(view.getUint32(40, true)).toBe(pcm.length * 2)
  })
  for (const wpm of [8, 12, 20, 30, 40, 50]) {
    test(`decodes ${wpm} WPM with automatic timing`, () => {
      const text = 'CQ CQ DE 9V1ABC TEST 599 K'
      const result = decodeSamples(synthesize(transmission(text, wpm), 650, 8000), 8000)
      expect(result.text.trim()).toBe(text)
      expect(Math.abs(result.wpm - wpm)).toBeLessThanOrEqual(3)
    })
  }
  test('silence stays silent', () => {
    const decoder = new CWDecoder(8000)
    for (let i = 0; i < 100; i++) decoder.process(new Float32Array(800))
    expect(decoder.finish().text).toBe('')
  })
  test('rejects noise without learning a fictitious fast sender', () => {
    for (const amplitude of [0.01, 0.1, 0.5]) {
      const random = randomSource(7741)
      const noise = Float32Array.from({ length: 8000 * 30 }, () => amplitude * Math.sqrt(-2 * Math.log(Math.max(1e-12, random()))) * Math.cos(2 * Math.PI * random()))
      expect(decodeSamples(noise, 8000).text).toBe('')
    }
  })
  test('follows rough human timing and stretched dahs without a word dictionary', () => {
    const text = 'CQ DE 9V1ABC UR RST 579 QTH SINGAPORE K'
    for (const challenge of [CHALLENGES[1], CHALLENGES[4]]) {
      const decoded = decodeSamples(makeSignal(text, challenge, 39181), 8000, { frequency: challenge.frequency, autoTune: false })
      expect(characterErrors(text, decoded.text).cer).toBeLessThanOrEqual(0.03)
    }
  })
  test('Farnsworth spacing and different input block sizes preserve copy', () => {
    const text = 'CQ TEST 73'
    const samples = synthesize(transmission(text, 25, 10), 800, 8000)
    for (const blockSize of [128, 256, 800]) {
      const decoder = new CWDecoder(8000, { wpm: 25, spacing: 10 })
      for (let offset = 0; offset < samples.length; offset += blockSize) decoder.process(samples.subarray(offset, offset + blockSize))
      expect(decoder.finish().text.trim()).toBe(text)
    }
  })
  test('commits older live copy without losing characters at the buffer boundary', () => {
    const text = Array(65).fill('CQ').join(' ')
    const decoded = decodeSamples(synthesize(transmission(text, 40), 650, 8000), 8000)
    expect(decoded.text.trim()).toBe(text)
  }, 15000)
})
