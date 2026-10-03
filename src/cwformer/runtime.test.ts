import { expect, test } from 'bun:test'
import { Tensor } from 'onnxruntime-web/wasm'
import { cacheTail, TOKENS, validateSettings } from './runtime'

test('CWformer trims the last 250 cache frames separately for every attention head', () => {
  const input = new Float32Array(4 * 253 * 64)
  for (let head = 0; head < 4; head++) {
    for (let frame = 0; frame < 253; frame++) input.fill(head * 1000 + frame, (head * 253 + frame) * 64, (head * 253 + frame + 1) * 64)
  }
  const output = cacheTail(new Tensor('float32', input, [1, 4, 253, 64]))
  expect(output.dims).toEqual([1, 4, 250, 64])
  for (let head = 0; head < 4; head++) {
    const data = output.data as Float32Array
    expect(data[head * 250 * 64]).toBe(head * 1000 + 3)
    expect(data[(head + 1) * 250 * 64 - 1]).toBe(head * 1000 + 252)
  }
  expect(cacheTail(output)).toBe(output)
  output.dispose()
})

test('browser model retains the released vocabulary and receiver limits', () => {
  expect(TOKENS.length).toBe(52)
  expect(TOKENS[0]).toBe('')
  expect(TOKENS[1]).toBe(' ')
  expect(TOKENS.at(-1)).toBe('CT')
  expect(() => validateSettings({ frequency: 650, bandwidth: 150 })).not.toThrow()
  for (const settings of [{ frequency: NaN, bandwidth: 150 }, { frequency: 1401, bandwidth: 150 }, { frequency: 650, bandwidth: 39 }]) {
    expect(() => validateSettings(settings)).toThrow()
  }
})
