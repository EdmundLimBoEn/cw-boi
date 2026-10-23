import { expect, test } from 'bun:test'
import { decodeText } from './morse'

test('Morse text treats inherited object properties as unknown symbols', () => {
  for (const token of ['constructor', 'toString', 'valueOf', 'hasOwnProperty', 'isPrototypeOf']) {
    expect(decodeText(`... ${token} / ---`)).toBe('S� O')
  }
  expect(decodeText('... --- ... / ...-.-')).toBe('SOS <SK>')
})
