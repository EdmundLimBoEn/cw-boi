"""Fine-tune the published acoustic model; independent validation seeds, no word LM."""

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF
from scipy.signal import hilbert, lfilter
from scipy.io import wavfile
from torch.utils.data import DataLoader, Dataset

from morse_synth.keying import KeyingConfig, render_events
from morse_synth.operator import OperatorConfig, build_events
from morseformer.core.tokenizer import encode, decode
from morseformer.features import FrontendConfig, extract_features
from server import load_model

SAMPLE_RATE = 8000
WINDOW = 48000
ALPHABET = np.array(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
WORDS = ['CQ', 'DE', 'UR', 'RST', 'QTH', 'NAME', 'RIG', 'PWR', 'ANT', 'TNX', 'FER', 'CALL', 'QSL', 'AGN', 'TEST', '73', '599', '559', '589', 'HR', 'WX', 'ES', 'BK', 'K']


def continuous_waveform(rng, clean):
    words = []
    for _ in range(10):
        if rng.random() < 0.45:
            words.append(str(rng.choice(WORDS)))
        else:
            words.append(''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789?/='), size=int(rng.integers(2, 9)))))
    source = ' '.join(words)
    wpm = rng.uniform(12, 40)
    unit = 1.2 / wpm
    dash = 3 if clean else rng.uniform(2.4, 4.8)
    mark_jitter = rng.uniform(0, 0.035 if clean else 0.35)
    gap_jitter = rng.uniform(0, 0.04 if clean else 0.4)
    intra = 1 if clean else rng.uniform(0.85, 1.3)
    char_gap = 3 if clean else rng.uniform(2.6, 3.8)
    word_gap = 7 if clean else rng.uniform(6, 10)
    swing = 0 if clean else rng.uniform(0, 0.3)
    phase = rng.uniform(0, 2 * math.pi)
    step = 0 if clean else rng.uniform(-0.3, 0.35)
    events, spans = [], []
    cursor = 0

    def add(on, duration):
        nonlocal cursor
        speed = (1 + swing * math.sin(cursor / SAMPLE_RATE / 2 + phase)) * (1 + step if cursor > SAMPLE_RATE * 8 else 1)
        duration *= speed * (1 + rng.uniform(-1, 1) * (mark_jitter if on else gap_jitter))
        if on and not clean and rng.random() < 0.12 and duration > 0.03:
            parts = [(True, 0.008), (False, 0.002), (True, duration - 0.01)]
        else:
            parts = [(on, duration)]
        events.extend(parts)
        cursor += sum(round(seconds * SAMPLE_RATE) for _, seconds in parts)

    for index, char in enumerate(source):
        if char == ' ':
            duration = word_gap * unit
            if not clean and rng.random() < 0.2:
                duration *= rng.uniform(1.4, 3)
            start = cursor
            add(False, duration)
            spans.append((char, start, cursor))
            continue
        if index and source[index - 1] != ' ':
            add(False, char_gap * unit)
        start = cursor
        for on, duration in build_events(char, OperatorConfig(wpm=wpm, dash_dot_ratio=dash)):
            add(on, duration if on else duration * intra)
        spans.append((char, start, cursor))

    rendered = render_events(events, freq=600 + (0 if clean else rng.uniform(-12, 12)), sample_rate=SAMPLE_RATE,
                             amplitude=0.4, tail_ms=0, keying=KeyingConfig(rise_ms=rng.uniform(2, 7)))
    offset = int(rng.integers(0, len(rendered) - WINDOW + 1))
    contained = [(char, start - offset, end - offset) for char, start, end in spans if start >= offset and end <= offset + WINDOW]
    # Partial edge characters are context, never labelled as complete characters.
    label = ''.join(char for char, _, _ in contained).strip()
    return rendered[offset:offset + WINDOW].copy(), label, contained


def case(seed, family, punctuation=False, continuous=False, background=None, return_audio=False):
    rng = np.random.default_rng(seed)
    clean = family == 'clean'
    chaotic = family == 'chaos'
    label = ''
    waveform = np.zeros(WINDOW, np.float32)
    if family != 'noise' and continuous:
        waveform, label, _ = continuous_waveform(rng, clean)
    elif family != 'noise':
        for attempt in range(80):
            length = int(rng.integers(2, 13 if attempt < 30 else 6))
            if rng.random() < 0.55:
                alphabet = np.array(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789?/=.')) if punctuation else ALPHABET
                label = ''.join(rng.choice(alphabet, size=length))
                if length > 5:
                    split = int(rng.integers(2, length - 1))
                    label = label[:split] + ' ' + label[split:]
            else:
                label = ' '.join(rng.choice(WORDS, size=int(rng.integers(1, 4))))
            wpm = rng.uniform(12, 40)
            jitter = rng.uniform(0, 0.035 if clean else 0.40)
            gap_jitter = rng.uniform(0, 0.04 if clean else 0.5)
            events = build_events(label, OperatorConfig(wpm=wpm, dash_dot_ratio=3 if clean else rng.uniform(2.4, 4.8)))
            step = 0 if clean else rng.uniform(-0.35, 0.4)
            swing = 0 if clean else rng.uniform(0, 0.3)
            shaped = []
            phase = rng.uniform(0, 2 * math.pi)
            for index, (on, duration) in enumerate(events):
                speed = (1 + swing * math.sin(index / 12 + phase)) * (1 + step if index > len(events) / 2 else 1)
                variation = 1 + rng.uniform(-1, 1) * (jitter if on else gap_jitter)
                if not clean and not on and duration > 5 * 1.2 / wpm and rng.random() < 0.25:
                    variation *= rng.uniform(1.5, 3)
                duration *= speed * variation
                if on and not clean and rng.random() < 0.12 and duration > 0.03:
                    shaped.extend([(True, 0.008), (False, 0.002), (True, duration - 0.010)])
                else:
                    shaped.append((on, duration))
            leading = rng.uniform(0.05, 0.45)
            if sum(duration for _, duration in shaped) + leading > 5.65:
                continue
            rendered = render_events([(False, leading)] + shaped, freq=600 + (0 if clean else rng.uniform(-12, 12)), sample_rate=SAMPLE_RATE,
                                     amplitude=0.4, tail_ms=100, keying=KeyingConfig(rise_ms=rng.uniform(2, 7)))
            assert len(rendered) <= WINDOW, 'Never truncate audio while retaining its full label.'
            waveform[:len(rendered)] = rendered
            break
        else:
            raise RuntimeError('Could not generate a complete labelled utterance.')
    time_axis = np.arange(WINDOW) / SAMPLE_RATE
    if not clean and family != 'noise':
        depth = rng.uniform(0.2, 0.85 if chaotic else 0.55)
        waveform *= (1 - depth * (0.5 + 0.5 * np.sin(time_axis * rng.uniform(0.7, 2.4) + rng.uniform(0, 6))))
        drift = rng.uniform(0, 28 if chaotic else 12) * np.sin(time_axis * rng.uniform(0.2, 0.8))
        waveform = np.real(hilbert(waveform) * np.exp(2j * np.pi * np.cumsum(drift) / SAMPLE_RATE)).astype(np.float32)
        if chaotic:
            waveform *= 1 - rng.uniform(0, 0.35) * (0.5 + 0.5 * np.sin(time_axis * rng.uniform(20, 45)))
    snr = rng.uniform(18, 35) if clean else rng.uniform(-8, 3) if chaotic else rng.uniform(5, 20)
    noise = rng.normal(0, 0.4 / math.sqrt(2) / 10 ** (snr / 20), WINDOW).astype(np.float32)
    if background is not None and family != 'noise':
        noise = background * (0.4 / math.sqrt(2) / 10 ** (snr / 20)) / max(1e-8, np.sqrt(np.mean(background ** 2)))
    if family == 'noise':
        noise *= rng.uniform(0.1, 4)
    waveform += noise
    if chaotic or (family == 'noise' and rng.random() < 0.5):
        impulses = np.zeros(WINDOW)
        positions = rng.integers(0, WINDOW, int(rng.poisson(6 * rng.uniform(0.3, 2))))
        impulses[positions] = rng.uniform(0.4, 2.5, len(positions))
        envelope = lfilter([1], [1, -0.998], impulses)
        crash = lfilter([0.15], [1, -0.85], rng.uniform(-1, 1, WINDOW))
        waveform += (envelope * crash * 4).astype(np.float32)
        if family != 'noise' and rng.random() < 0.55:
            other = ''.join(rng.choice(ALPHABET, size=15))
            offset = rng.choice([-1, 1]) * rng.uniform(100, 260)
            qrm = render_events(build_events(other, OperatorConfig(wpm=rng.uniform(20, 42))), freq=600 + offset,
                                sample_rate=SAMPLE_RATE, amplitude=rng.uniform(0.15, 0.5))
            count = min(len(qrm), WINDOW)
            waveform[:count] += qrm[:count]
    if return_audio:
        return waveform, label
    features = extract_features(waveform, SAMPLE_RATE, FrontendConfig(tone_freq=600, bandwidth=100))
    return features, label


class TrainingData(Dataset):
    def __init__(self, length, seed, hard_fraction=0.25, continuous_fraction=0, noise_recordings=()):
        self.length, self.seed, self.hard_fraction = length, seed, hard_fraction
        self.continuous_fraction = continuous_fraction
        self.backgrounds = []
        for path in noise_recordings:
            rate, samples = wavfile.read(path)
            if rate != SAMPLE_RATE or samples.ndim != 1 or len(samples) < WINDOW:
                raise ValueError(f'{path}: background must be mono 8 kHz and at least six seconds.')
            samples = samples.astype(np.float32)
            samples /= max(1, np.max(np.abs(samples)))
            if not np.isfinite(samples).all():
                raise ValueError(f'{path}: invalid background samples.')
            self.backgrounds.append(samples)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        choice = np.random.default_rng(self.seed + index).random()
        family = 'noise' if choice < 0.15 else 'clean' if choice < 0.4 else 'rough' if choice < 1 - self.hard_fraction else 'chaos'
        continuous = np.random.default_rng(self.seed + index + 900000000).random() < self.continuous_fraction
        background = None
        rng = np.random.default_rng(self.seed + index + 800000000)
        if self.backgrounds and family in ('rough', 'chaos') and rng.random() < 0.5:
            recording = self.backgrounds[int(rng.integers(len(self.backgrounds)))]
            start = int(rng.integers(len(recording) - WINDOW + 1))
            background = recording[start:start + WINDOW]
        features, text = case(self.seed + index, family, punctuation=True, continuous=continuous, background=background)
        return torch.from_numpy(features), torch.tensor(encode(text), dtype=torch.long)


def collate(items):
    features, tokens = zip(*items)
    lengths = torch.tensor([len(item) for item in tokens], dtype=torch.long)
    targets = torch.zeros((len(items), max(1, int(lengths.max()))), dtype=torch.long)
    for i, item in enumerate(tokens):
        targets[i, :len(item)] = item
    return torch.stack(features), targets, lengths


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ac in enumerate(a, 1):
        row = [i]
        for j, bc in enumerate(b, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (ac != bc)))
        previous = row
    return previous[-1]


@torch.inference_mode()
def evaluate(model, device, count=48, seed=2000000):
    model.eval()
    totals = {family: {'edits': 0, 'characters': 0, 'clips': 0, 'false_characters': 0} for family in ('clean', 'rough', 'chaos', 'noise')}
    examples = []
    for i in range(count):
        family = list(totals)[i % 4]
        features, expected = case(seed + i, family)
        x = torch.from_numpy(features).unsqueeze(0).to(device)
        lengths = torch.tensor([len(features)], device=device)
        tokens = model.greedy_rnnt_decode(x, lengths, confidence_threshold=0.6, digit_threshold=0.9)[0]
        actual = decode(tokens)
        record = totals[family]
        record['clips'] += 1
        record['edits'] += edit_distance(expected, actual)
        record['characters'] += len(expected)
        if not expected:
            record['false_characters'] += len(actual)
        examples.append({'family': family, 'expected': expected, 'actual': actual})
    for record in totals.values():
        record['cer'] = record['edits'] / max(1, record['characters'])
    return {'metrics': totals, 'examples': examples, 'seed': seed}


def acceptable(candidate, baseline):
    c, b = candidate['metrics'], baseline['metrics']
    return (c['clean']['cer'] <= b['clean']['cer'] + 0.02
            and c['rough']['cer'] <= b['rough']['cer'] + 0.02
            and c['noise']['false_characters'] <= b['noise']['false_characters']
            and c['chaos']['cer'] < b['chaos']['cer'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--eval-every', type=int, default=200)
    parser.add_argument('--eval-count', type=int, default=48)
    parser.add_argument('--eval-seed', type=int, default=2000000)
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--seed', type=int, default=110001)
    parser.add_argument('--output', type=Path, default=Path('models/rough-fist-v1'))
    parser.add_argument('--checkpoint')
    parser.add_argument('--hard-fraction', type=float, default=0.25, help='Fraction of training examples with combined radio distortions (0–0.6).')
    parser.add_argument('--continuous-fraction', type=float, default=0, help='Fraction of nonempty examples cropped from ongoing transmissions.')
    parser.add_argument('--save-checkpoints', action='store_true', help='Retain each evaluation checkpoint for independent continuous-audio comparison.')
    parser.add_argument('--noise-recordings', nargs='*', default=[], help='Training-only HF recordings mixed under labelled CW; never used as empty targets.')
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1 or args.eval_count < 4:
        parser.error('Use positive step, batch, and evaluation counts (at least 4 evaluation clips).')
    if not 0 <= args.hard_fraction <= 0.6:
        parser.error('Hard fraction must be between 0 and 0.6.')
    if not 0 <= args.continuous_fraction <= 1:
        parser.error('Continuous fraction must be between 0 and 1.')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    model = load_model(args.checkpoint).to(args.device)
    metadata = {'arguments': vars(args) | {'output': str(args.output)}, 'torch': str(torch.__version__),
                'torchaudio': str(torchaudio.__version__), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'noise_recordings': [{'path': path, 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()} for path in args.noise_recordings],
                'hardware': torch.cuda.get_device_name(0) if args.device == 'cuda' else 'CPU'}
    (args.output / 'run.json').write_text(json.dumps(metadata, indent=2))
    baseline = evaluate(model, args.device, args.eval_count, args.eval_seed)
    (args.output / 'baseline.json').write_text(json.dumps(baseline, indent=2))
    print(json.dumps({'event': 'baseline', 'metrics': baseline['metrics']}), flush=True)
    if args.evaluate_only:
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    data = DataLoader(TrainingData(args.steps * args.batch_size, args.seed, args.hard_fraction, args.continuous_fraction, args.noise_recordings), batch_size=args.batch_size,
                      num_workers=args.workers, collate_fn=collate, pin_memory=args.device == 'cuda')
    started = time.monotonic()
    best = baseline
    for step, (features, tokens, n_tokens) in enumerate(data, 1):
        model.train()
        features, tokens, n_tokens = features.to(args.device), tokens.to(args.device), n_tokens.to(args.device)
        lengths = torch.full((len(features),), features.shape[1], dtype=torch.long, device=args.device)
        out = model(features, tokens, lengths=lengths)
        enc_lengths = out['enc_lengths']
        targets = torch.cat([tokens[i, :int(length)] for i, length in enumerate(n_tokens)])
        ctc = ctc_loss(out['ctc_log_probs'].transpose(0, 1), targets, enc_lengths, n_tokens)
        nonempty = (n_tokens > 0).nonzero(as_tuple=True)[0]
        rnnt = torch.zeros((), device=args.device)
        if len(nonempty):
            max_u = int(n_tokens[nonempty].max())
            rnnt = AF.rnnt_loss(out['joint_logits'][nonempty, :, :max_u + 1, :].float().contiguous(), tokens[nonempty, :max_u].int().contiguous(),
                               enc_lengths[nonempty].int(), n_tokens[nonempty].int(), blank=0, reduction='mean')
        empty = (n_tokens == 0).nonzero(as_tuple=True)[0]
        blank_loss = torch.zeros((), device=args.device)
        if len(empty):
            # Supervise silence directly; torchaudio's zero-target RNN-T path is unstable.
            blank_loss = -out['joint_logits'][empty, :, 0, :].log_softmax(-1)[..., 0].mean()
        loss = 0.3 * ctc + 0.7 * (rnnt * len(nonempty) + blank_loss * len(empty)) / len(features)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite loss at step {step}; checkpoint was not promoted.')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if step % 20 == 0 or step == 1:
            print(json.dumps({'event': 'train', 'step': step, 'loss': float(loss.detach()), 'seconds': round(time.monotonic() - started)}), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            candidate = evaluate(model, args.device, args.eval_count, args.eval_seed)
            report = {'step': step, 'accepted': acceptable(candidate, baseline), **candidate}
            (args.output / f'eval-{step}.json').write_text(json.dumps(report, indent=2))
            state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save({'model': state, 'step': step, 'config': vars(args) | {'output': str(args.output)}, 'metrics': candidate['metrics']}, args.output / 'last.pt')
            if args.save_checkpoints:
                torch.save({'model': state, 'step': step, 'metrics': candidate['metrics']}, args.output / f'step-{step}.pt')
            if report['accepted'] and candidate['metrics']['chaos']['cer'] < best['metrics']['chaos']['cer']:
                torch.save({'model': state, 'step': step, 'metrics': candidate['metrics']}, args.output / 'best.pt')
                best = copy.deepcopy(candidate)
            print(json.dumps({'event': 'evaluation', 'step': step, 'accepted': report['accepted'], 'metrics': candidate['metrics']}), flush=True)
    print(json.dumps({'event': 'complete', 'best': best['metrics'], 'seconds': round(time.monotonic() - started)}), flush=True)


if __name__ == '__main__':
    main()
