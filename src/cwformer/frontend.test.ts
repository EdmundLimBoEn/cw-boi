import { describe, expect, test } from 'bun:test'
import { Frontend, MelFrontend } from './frontend'

const window = Float32Array.from({ length: 400 }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / 400))
const basis = Float32Array.from({ length: 40 * 201 }, (_, i) => {
  const mel = Math.floor(i / 201), bin = i % 201
  return Math.max(0, 1 - Math.abs(bin - mel * 5) / 3)
})

function directMel(audio: Float32Array) {
  const joined = new Float32Array(200 + audio.length)
  joined.set(audio, 200)
  const count = Math.max(0, Math.floor((joined.length - 400) / 160) + 1)
  const result = new Float32Array(count * 40)
  for (let frame = 0; frame < count; frame++) {
    const power = new Float64Array(201)
    for (let bin = 0; bin < 201; bin++) {
      let real = 0, imag = 0
      for (let i = 0; i < 400; i++) {
        const value = Math.fround(joined[frame * 160 + i] * window[i]), angle = -2 * Math.PI * bin * i / 400
        real += value * Math.cos(angle)
        imag += value * Math.sin(angle)
      }
      power[bin] = real * real + imag * imag
    }
    for (let mel = 0; mel < 40; mel++) {
      let energy = 0
      for (let bin = 0; bin < 201; bin++) energy += power[bin] * basis[mel * 201 + bin]
      result[frame * 40 + mel] = Math.log(energy + 1e-6)
    }
  }
  return result
}

describe('CWformer browser audio frontend', () => {
  test('matches a direct 400-point DFT with float32 window multiplication', () => {
    const audio = Float32Array.from({ length: 520 }, (_, i) => Math.sin(i * 0.13) * 0.3 + Math.cos(i * 0.29) * 0.2)
    const mel = new MelFrontend(window, basis).process(audio)
    const reference = directMel(audio)
    expect(mel.frames).toBe(3)
    expect(mel.data.length).toBe(reference.length)
    for (let i = 0; i < reference.length; i++) expect(Math.abs(mel.data[i] - reference[i])).toBeLessThan(1e-5)
  })

  test('retains exact frame overlap across empty and short streaming chunks', () => {
    const audio = Float32Array.from({ length: 1047 }, (_, i) => Math.sin(i * 0.53) * 0.17)
    const expected = new MelFrontend(window, basis).process(audio)
    const streaming = new MelFrontend(window, basis)
    expect(streaming.process(new Float32Array()).frames).toBe(0)
    const pieces = [streaming.process(audio.slice(0, 199)), streaming.process(audio.slice(199, 409)),
      streaming.process(audio.slice(409, 446)), streaming.process(audio.slice(446))]
    expect(pieces[0].frames).toBe(0)
    expect(pieces.reduce((sum, piece) => sum + piece.frames, 0)).toBe(expected.frames)
    expect(pieces.flatMap(piece => [...piece.data])).toEqual([...expected.data])
  })

  test('preserves silence and uses the production 2x sample rate and block peak', () => {
    const frontend = new Frontend(710, 150)
    expect([...frontend.process(new Float32Array(200))].every(value => value === 0)).toBe(true)
    const audio = Float32Array.from({ length: 256 }, (_, i) => 0.1 * Math.sin(2 * Math.PI * 710 * i / 8000))
    const output = frontend.process(audio)
    expect(output.length).toBe(512)
    expect(Math.max(...output.map(Math.abs))).toBeCloseTo(0.7, 6)
    expect([...output].every(Number.isFinite)).toBe(true)
    frontend.frequency = 715
    expect([...frontend.process(audio)].every(Number.isFinite)).toBe(true)
  })
})
