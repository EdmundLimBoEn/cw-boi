export const MORSE: Record<string, string> = {
  A: '.-', B: '-...', C: '-.-.', D: '-..', E: '.', F: '..-.', G: '--.',
  H: '....', I: '..', J: '.---', K: '-.-', L: '.-..', M: '--', N: '-.',
  O: '---', P: '.--.', Q: '--.-', R: '.-.', S: '...', T: '-', U: '..-',
  V: '...-', W: '.--', X: '-..-', Y: '-.--', Z: '--..',
  '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-',
  '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.',
  '.': '.-.-.-', ',': '--..--', '?': '..--..', "'": '.----.', '!': '-.-.--',
  '/': '-..-.', '(': '-.--.', ')': '-.--.-', '&': '.-...', ':': '---...',
  ';': '-.-.-.', '=': '-...-', '+': '.-.-.', '-': '-....-', '_': '..--.-',
  '"': '.-..-.', '$': '...-..-', '@': '.--.-.', É: '..-..',
  '<AR>': '.-.-.', '<AS>': '.-...', '<SK>': '...-.-', '<BT>': '-...-',
  '<KN>': '-.--.', '<SOS>': '...---...', '<HH>': '........',
}

export const ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'.split('')
export const REVERSE = Object.fromEntries(Object.entries(MORSE).map(([a, b]) => [b, a]))
export type Token = { text: string; code: string; index: number }
export type Tone = { start: number; end: number; index: number; symbol: string }
export type Transmission = { tones: Tone[]; duration: number; tokens: Token[]; unsupported: string[] }
export type SenderSettings = { wpm: number; spacing: number; frequency: number; volume: number }
export const DEFAULT_SENDER: SenderSettings = { wpm: 20, spacing: 20, frequency: 650, volume: 0.35 }

export function tokenize(text: string) {
  const tokens: Token[] = []
  const unsupported = new Set<string>()
  const pattern = /<[^>]*>|\s+|./gu
  for (const match of text.toUpperCase().matchAll(pattern)) {
    const char = match[0]
    if (/^\s+$/.test(char)) {
      if (tokens.length && tokens.at(-1)?.text !== ' ') tokens.push({ text: ' ', code: '/', index: match.index! })
    } else if (MORSE[char]) {
      tokens.push({ text: char, code: MORSE[char], index: match.index! })
    } else unsupported.add(char)
  }
  while (tokens.at(-1)?.text === ' ') tokens.pop()
  return { tokens, unsupported: [...unsupported] }
}

export function transmission(text: string, wpm: number, spacing = wpm): Transmission {
  if (!Number.isFinite(wpm) || !Number.isFinite(spacing) || wpm < 5 || wpm > 60 || spacing < 5 || spacing > wpm) {
    throw new Error('Use a character speed of 5–60 WPM and a spacing speed between 5 WPM and the character speed.')
  }
  if (text.length > 1000) throw new Error('Keep each transmission under 1,000 characters.')
  const { tokens, unsupported } = tokenize(text)
  const dit = 1.2 / wpm
  // PARIS is 31 mark/intra-character units and 19 spacing units per word.
  const gapUnit = (60 / spacing - 31 * dit) / 19
  const tones: Tone[] = []
  let time = 0.15
  tokens.forEach((token, index) => {
    if (token.text === ' ') return
    token.code.split('').forEach((symbol, element) => {
      const length = dit * (symbol === '-' ? 3 : 1)
      tones.push({ start: time, end: time + length, index, symbol })
      time += length + (element < token.code.length - 1 ? dit : 0)
    })
    if (index < tokens.length - 1) time += gapUnit * (tokens[index + 1].text === ' ' ? 7 : 3)
  })
  if (time > 600) throw new Error('This transmission is over 10 minutes. Shorten the message or increase the speed.')
  return { tones, duration: tones.length ? time + Math.max(0.5, 7 * gapUnit) : 0, tokens, unsupported }
}

export function encodeText(text: string) {
  return tokenize(text).tokens.map(token => token.code).join(' ')
}

export function decodeText(text: string) {
  const normalized = text.trim().replace(/[·•]/g, '.').replace(/[−–—_]/g, '-').replace(/\s*[|/]\s*|\s{3,}/g, ' / ')
  return normalized.split(/\s+/).filter(Boolean).map(code => code === '/' ? ' ' : REVERSE[code] ?? '�').join('')
}

export function synthesize(plan: Transmission, frequency: number, sampleRate = 22050, volume = 0.6) {
  if (!Number.isFinite(frequency) || frequency < 200 || frequency > 1500) throw new Error('Tone must be between 200 and 1,500 Hz.')
  const samples = new Float32Array(Math.ceil(plan.duration * sampleRate))
  const ramp = Math.round(0.004 * sampleRate)
  for (const tone of plan.tones) {
    const start = Math.round(tone.start * sampleRate)
    const end = Math.min(samples.length, Math.round(tone.end * sampleRate))
    for (let i = start; i < end; i++) {
      const edge = Math.min(1, (i - start) / ramp, (end - 1 - i) / ramp)
      samples[i] = volume * (0.5 - 0.5 * Math.cos(Math.PI * edge)) * Math.sin(2 * Math.PI * frequency * i / sampleRate)
    }
  }
  return samples
}

export function wavBytes(samples: Float32Array, sampleRate = 22050) {
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(buffer)
  const write = (offset: number, text: string) => [...text].forEach((letter, i) => view.setUint8(offset + i, letter.charCodeAt(0)))
  write(0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true); write(8, 'WAVE')
  write(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true)
  view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true); view.setUint16(34, 16, true); write(36, 'data'); view.setUint32(40, samples.length * 2, true)
  for (let i = 0; i < samples.length; i++) view.setInt16(44 + i * 2, Math.round(Math.max(-1, Math.min(1, samples[i])) * 32767), true)
  return buffer
}
