import { transmission, type Tone } from './morse'

export type Challenge = { id: string; name: string; description: string; wpm: number; frequency: number; snr: number; jitter: number; drift: number; fading: number; dashRatio?: number; swing?: number; interference?: number; gapJitter?: number; speedStep?: number; hesitation?: number; impulses?: number; flutter?: number; bounce?: number }
export const CHALLENGES: Challenge[] = [
  { id: 'clean', name: 'A clear channel', description: 'A clean, machine-keyed signal. Start here.', wpm: 20, frequency: 650, snr: 30, jitter: 0, drift: 0, fading: 0 },
  { id: 'human', name: 'The human touch', description: 'Uneven spacing, stretched dahs, and 20% timing variation.', wpm: 18, frequency: 720, snr: 15, jitter: 0.2, drift: 0, fading: 0, dashRatio: 3.6, swing: 0.2 },
  { id: 'fading', name: 'Somewhere far away', description: 'A wandering carrier with deep, slow signal fading.', wpm: 23, frequency: 580, snr: 5, jitter: 0.1, drift: 18, fading: 0.75 },
  { id: 'crowded', name: 'Company on the band', description: 'A nearby interfering station and a noisy background.', wpm: 25, frequency: 800, snr: 3, jitter: 0.12, drift: 4, fading: 0.2, interference: 180 },
  { id: 'rough', name: 'A very rough fist', description: '35% mark jitter, 45% gap jitter, long dahs, hesitations, and a sudden speed change.', wpm: 19, frequency: 680, snr: 10, jitter: 0.35, gapJitter: 0.45, drift: 8, fading: 0.25, dashRatio: 4.1, swing: 0.3, speedStep: 0.35, hesitation: 0.45, bounce: 0.3 },
  { id: 'chaos', name: 'Everything, all at once', description: 'Rough keying buried in noise, static crashes, flutter, fading, and another station.', wpm: 22, frequency: 620, snr: -5, jitter: 0.3, gapJitter: 0.4, drift: 23, fading: 0.65, dashRatio: 3.9, swing: 0.25, speedStep: -0.25, hesitation: 0.3, interference: 125, impulses: 1.5, flutter: 0.25, bounce: 0.15 },
]

export function randomSource(seed: number) {
  let state = seed | 0
  return () => {
    state ^= state << 13; state ^= state >>> 17; state ^= state << 5
    return (state >>> 0) / 4294967296
  }
}

export function makeSignal(text: string, challenge: Challenge, seed = 73, sampleRate = 8000) {
  const random = randomSource(seed)
  const base = transmission(text, challenge.wpm)
  const tones: Tone[] = []
  let end = 0, previousEnd = 0
  for (let i = 0; i < base.tones.length; i++) {
    const tone = base.tones[i]
    const speedCurve = (1 + (challenge.swing ?? 0) * Math.sin(i / 17)) * (i > base.tones.length / 2 ? 1 + (challenge.speedStep ?? 0) : 1)
    let gap = (tone.start - previousEnd) * speedCurve * (1 + (challenge.gapJitter ?? challenge.jitter) * (random() * 2 - 1))
    if (challenge.hesitation && tone.start - previousEnd > 1.2 / challenge.wpm * 5 && random() < challenge.hesitation) gap *= 2 + random() * 3
    let duration = (tone.end - tone.start) * speedCurve * (1 + challenge.jitter * (random() * 2 - 1))
    if (tone.symbol === '-') duration *= (challenge.dashRatio ?? 3) / 3
    tones.push({ ...tone, start: end + gap, end: end + gap + duration })
    previousEnd = tone.end
    end += gap + duration
  }
  const samples = new Float32Array(Math.ceil((end + 1) * sampleRate))
  const amplitude = 0.4
  // SNR is keyed carrier RMS versus broadband Gaussian noise RMS at this sample rate.
  const noiseRms = amplitude / Math.sqrt(2) / 10 ** (challenge.snr / 20)
  let toneIndex = 0, phase = 0
  const interfering = challenge.interference ? transmission('CQ CQ DE W9XYZ UR 599 599 QSL? CQ CQ DE W9XYZ K', 31).tones : []
  let interfererIndex = 0, crash = 0, crashState = 0
  for (let i = 0; i < samples.length; i++) {
    const time = i / sampleRate
    while (toneIndex < tones.length && tones[toneIndex].end < time) toneIndex++
    const tone = tones[toneIndex]
    const u1 = Math.max(1e-12, random()), u2 = random()
    let value = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * noiseRms
    phase += 2 * Math.PI * (challenge.frequency + challenge.drift * Math.sin(time / 3)) / sampleRate
    if (tone && time >= tone.start && time < tone.end) {
      const edge = Math.min(1, (time - tone.start) / 0.004, (tone.end - time) / 0.004)
      const fade = (1 - challenge.fading * (0.5 + 0.5 * Math.sin(time * 1.2))) * (1 - (challenge.flutter ?? 0) * (0.5 + 0.5 * Math.sin(time * 29)))
      const bounced = challenge.bounce && (toneIndex * 0.618 % 1) < challenge.bounce && time - tone.start > 0.009 && time - tone.start < 0.012
      value += (bounced ? 0 : amplitude) * fade * (0.5 - 0.5 * Math.cos(Math.PI * edge)) * Math.sin(phase)
    }
    if (challenge.interference) {
      const cycle = time % (interfering.at(-1)!.end + 0.8)
      if (cycle < (interfering[interfererIndex]?.start ?? 0) - 0.2) interfererIndex = 0
      while (interfererIndex < interfering.length && interfering[interfererIndex].end < cycle) interfererIndex++
      const other = interfering[interfererIndex]
      if (other && cycle >= other.start) value += 0.35 * Math.sin(2 * Math.PI * (challenge.frequency + challenge.interference) * time)
    }
    if (challenge.impulses) {
      if (random() < challenge.impulses / sampleRate) crash = 1.4 + random() * 2
      crash *= 0.998
      crashState = crashState * 0.85 + (random() * 2 - 1) * 0.15
      value += crash * crashState * 4
    }
    samples[i] = value
  }
  return samples
}

export function characterErrors(expected: string, actual: string) {
  const a = expected.trim().toUpperCase(), b = actual.trim().toUpperCase()
  let previous = Array.from({ length: b.length + 1 }, (_, i) => i)
  for (let i = 1; i <= a.length; i++) {
    const row = [i]
    for (let j = 1; j <= b.length; j++) row[j] = Math.min(row[j - 1] + 1, previous[j] + 1, previous[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1))
    previous = row
  }
  const errors = previous[b.length]
  return { errors, cer: a.length ? errors / a.length : b.length ? 1 : 0 }
}
