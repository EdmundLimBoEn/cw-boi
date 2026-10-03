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
    for (const amplitude of [0.00001, 0.0001, 0.01, 0.1, 0.5]) {
      const random = randomSource(7741)
      const noise = Float32Array.from({ length: 8000 * 30 }, () => amplitude * Math.sqrt(-2 * Math.log(Math.max(1e-12, random()))) * Math.cos(2 * Math.PI * random()))
      expect(decodeSamples(noise, 8000).text).toBe('')
    }
  })
  test('acquires a quiet high-SNR carrier both initially and after silence', () => {
    for (const lead of [0, 16000]) {
      const signal = makeSignal('CQ TEST 73', { ...CHALLENGES[0], frequency: 900 }, 1349761).map(value => value * 0.0001)
      const samples = new Float32Array(lead + signal.length)
      samples.set(signal, lead)
      expect(Math.abs(decodeSamples(samples, 8000).frequency - 900)).toBeLessThan(3)
    }
  })
  test('retains an active station through key-up gaps and acquires a new carrier after it stops', () => {
    const target = makeSignal('CQ DE W1ABC UR RST 599 TEST 73', { ...CHALLENGES[0], frequency: 800 }, 68113)
    const neighbor = synthesize(transmission('CQ CQ DE K9XYZ UR 579 K CQ CQ DE K9XYZ UR 579 K', 25), 1000, 8000, 0.52)
    for (let i = 16000; i < target.length; i++) target[i] += neighbor[i - 16000] ?? 0
    const decoder = new CWDecoder(8000, { bandwidth: 150 })
    for (let at = 0; at < target.length; at += 160) {
      const reading = decoder.process(target.subarray(at, at + 160))
      if (at >= 16000) expect(Math.abs(reading.frequency - 800)).toBeLessThanOrEqual(8)
    }
    decoder.process(new Float32Array(24000))
    const next = synthesize(transmission('TEST 599 K', 20), 500, 8000)
    for (let at = 0; at < 8000; at += 160) decoder.process(next.subarray(at, at + 160))
    expect(Math.abs(decoder.frequency - 500)).toBeLessThan(4)
    decoder.configure({ autoTune: false, frequency: 720 })
    decoder.process(next.subarray(8000))
    expect(decoder.frequency).toBe(720)
  })
  test('rejects colored receiver noise and still acquires a carrier above it', () => {
    for (const seed of [915, 919]) {
      const random = randomSource(seed)
      let colored = 0
      const noise = Float32Array.from({ length: 8000 * 30 }, () => {
        const white = 0.3 * Math.sqrt(-2 * Math.log(Math.max(1e-12, random()))) * Math.cos(2 * Math.PI * random())
        colored = colored * 0.9 + white * 0.1
        return colored * 3
      })
      expect(decodeSamples(noise, 8000).text).toBe('')
      const message = synthesize(transmission('CQ TEST 73', 20), 850, 8000)
      for (let i = 0; i < message.length; i++) message[i] += noise[i]
      expect(decodeSamples(message, 8000).text.trim()).toBe('CQ TEST 73')
    }
  })
  test('rejects static crashes and a steady unkeyed carrier', () => {
    const random = randomSource(855490107)
    let crash = 0
    const staticNoise = Float32Array.from({ length: 8000 * 20 }, () => {
      const white = Math.sqrt(-2 * Math.log(Math.max(1e-12, random()))) * Math.cos(2 * Math.PI * random())
      if (random() < 1.5 / 8000) crash = 3
      crash *= 0.998
      return white * (0.03 + crash)
    })
    expect(decodeSamples(staticNoise, 8000, { frequency: 584, autoTune: false }).text).toBe('')
    const carrier = Float32Array.from({ length: 8000 * 10 }, (_, i) => 0.4 * Math.sin(2 * Math.PI * 650 * i / 8000))
    expect(decodeSamples(carrier, 8000).text).toBe('')
    const message = synthesize(transmission('CQ TEST', 20), 650, 8000)
    const resumed = new Float32Array(carrier.length + 8000 + message.length)
    resumed.set(carrier); resumed.set(message, carrier.length + 8000)
    expect(decodeSamples(resumed, 8000).text.trim()).toBe('CQ TEST')
  })
  test('rejects noise shaped by a narrow receiver filter while retaining keyed audio', () => {
    const random = randomSource(82619), noise = new Float32Array(8000 * 20)
    let i1 = 0, q1 = 0, i2 = 0, q2 = 0
    for (let n = 0; n < noise.length; n++) {
      i1 = 0.94 * i1 + 0.06 * (random() * 2 - 1); q1 = 0.94 * q1 + 0.06 * (random() * 2 - 1)
      i2 = 0.94 * i2 + 0.06 * i1; q2 = 0.94 * q2 + 0.06 * q1
      noise[n] = 0.1 * (i2 * Math.cos(2 * Math.PI * 800 * n / 8000) + q2 * Math.sin(2 * Math.PI * 800 * n / 8000))
    }
    expect(decodeSamples(noise, 8000, { frequency: 800, autoTune: false }).text).toBe('')
    const message = synthesize(transmission('CQ TEST 73', 20), 800, 8000)
    for (let n = 0; n < message.length; n++) message[n] = message[n] * 0.1 + noise[n]
    expect(decodeSamples(message, 8000, { frequency: 800, autoTune: false }).text.trim().startsWith('CQ TEST 73')).toBe(true)
  })
  test('preserves the first dot when receiver noise starts after digital silence', () => {
    for (const snr of [15, 30]) {
      const signal = makeSignal('AFTER WATER TEST', { ...CHALLENGES[0], frequency: 650, snr }, 1004519824)
      const delayed = new Float32Array(signal.length + 8000)
      delayed.set(signal, 8000)
      expect(decodeSamples(delayed, 8000, { autoTune: false }).text.trim()).toBe('AFTER WATER TEST')
    }
  })
  test('keeps a weak carrier inside the filter when manual tuning is slightly offset', () => {
    const text = 'CQ TEST 73 DE 9V1ABC'
    for (const offset of [-30, -18, 18, 30]) {
      const signal = makeSignal(text, { ...CHALLENGES[0], frequency: 650 + offset, snr: 0, fading: 0.4 }, 83761)
      expect(decodeSamples(signal, 8000, { frequency: 650, autoTune: false }).text.trim()).toBe(text)
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
