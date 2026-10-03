"""Adapt CWformer with generated human timing and optional recorded interference."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfilt
from torch.utils.data import DataLoader, Dataset

import train as synth
from cwformer_engine import Frontend

REVISION = 'ef6ac7ca75b20833c811ea9ebf2bde1fa139fa70'
UPSTREAM = Path(os.environ.get('CWFORMER_SOURCE', Path(__file__).resolve().parents[1] / '.research/cwformer'))
if not (UPSTREAM / 'neural_decoder').is_dir():
    raise RuntimeError(f'Download parsimo2010/CWformer revision {REVISION} to .research/cwformer before training.')
sys.path.insert(0, str(UPSTREAM))
from neural_decoder.conformer import ConformerConfig
from neural_decoder.cwformer import CWFormer, CWFormerConfig
from neural_decoder.mel_frontend import MelFrontendConfig
from vocab import char_to_idx


def load(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    saved = checkpoint['model_config']
    config = CWFormerConfig(
        mel=MelFrontendConfig(spec_augment=False),
        conformer=ConformerConfig(
            d_model=saved['d_model'], n_heads=saved['n_heads'], n_layers=saved['n_layers'],
            d_ff=saved['d_ff'], conv_kernel=saved['conv_kernel'], max_cache_len=250, dropout=.05))
    model = CWFormer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model, saved


def prepare(waveform, frequency=600, bandwidth=150, frontend='legacy'):
    if frontend == 'causal':
        processor = Frontend(frequency, bandwidth)
        return np.concatenate([processor.process(waveform[at:at + 4000])
                               for at in range(0, len(waveform), 4000)])
    if frontend != 'legacy':
        raise ValueError(f'Unknown frontend: {frontend}')
    time_axis = np.arange(len(waveform)) / 8000
    mixed = waveform * np.exp(-2j * np.pi * frequency * time_axis) * 2
    baseband = sosfilt(butter(3, bandwidth / 2, fs=8000, output='sos'), mixed)
    shifted = (baseband * np.exp(2j * np.pi * 650 * time_axis)).real.astype(np.float32)
    return resample_poly(shifted, 2, 1).astype(np.float32)


class Data(Dataset):
    def __init__(self, count, seed, noise_paths, frontend='legacy', short_gap_fraction=0, element_gap_range=(.25, 1)):
        self.count, self.seed = count, seed
        self.frontend = frontend
        self.short_gap_fraction, self.element_gap_range = short_gap_fraction, element_gap_range
        self.backgrounds = []
        for path in noise_paths:
            rate, audio = wavfile.read(path)
            if rate != 8000 or audio.ndim != 1 or len(audio) <= 48000 or not np.isfinite(audio).all():
                raise ValueError(f'{path}: background must be finite mono 8 kHz audio longer than six seconds')
            self.backgrounds.append(audio.astype(np.float32))

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + index)
        weighting_rng = np.random.default_rng(self.seed + index + 400000000)
        pure_noise = rng.random() < .2
        waveforms, labels = [], []
        for part in range(2):
            family = 'noise' if pure_noise else str(rng.choice(['clean', 'rough', 'chaos'], p=[.3, .4, .3]))
            background = None
            if self.backgrounds and rng.random() < .5:
                recording = self.backgrounds[int(rng.integers(len(self.backgrounds)))]
                start = int(rng.integers(len(recording) - 48000 + 1))
                background = recording[start:start + 48000]
            # Recorded interference is never used as empty-transcript ground truth.
            gap_scale = (weighting_rng.uniform(*self.element_gap_range)
                         if family != 'noise' and weighting_rng.random() < self.short_gap_fraction else 1)
            waveform, label = synth.case(
                self.seed + index * 17 + part, family, punctuation=True,
                continuous=False, background=background, return_audio=True, element_gap_scale=gap_scale)
            waveforms.append(waveform)
            labels.append(label)
        joined = np.zeros(112000, np.float32)
        offset = int(rng.integers(0, 8001))
        joined[offset:offset + 48000] = waveforms[0]
        joined[offset + 56000:offset + 104000] = waveforms[1]
        audio = prepare(joined, bandwidth=float(rng.uniform(100, 220)), frontend=self.frontend)
        if self.frontend == 'legacy':
            audio *= rng.uniform(.4, .9) / max(np.max(np.abs(audio)), 1e-6)
        text = ' '.join(label for label in labels if label)
        text = f' {text} ' if text else ''
        return torch.from_numpy(audio), torch.tensor([char_to_idx[char] for char in text], dtype=torch.int64)


def collate(items):
    waveforms, labels = zip(*items)
    return (torch.stack(waveforms), torch.cat(labels),
            torch.tensor([len(label) for label in labels], dtype=torch.int64))


def ctc_loss(probabilities, targets, output_lengths, target_lengths):
    losses = torch.nn.functional.ctc_loss(
        probabilities, targets, output_lengths, target_lengths,
        blank=0, reduction='none', zero_infinity=False)
    normalizer = torch.where(target_lengths > 0, target_lengths, output_lengths)
    return (losses / normalizer).mean()


def self_check():
    from unittest.mock import patch

    for index in range(12):
        audio, targets = Data(12, 67201911, [])[index]
        assert audio.shape == (224000,) and audio.isfinite().all()
        if len(targets):
            assert targets[0] == char_to_idx[' '] and targets[-1] == char_to_idx[' ']
    for frames in (100, 200):
        probabilities = torch.tensor([.8, .2]).log().repeat(frames, 1, 1)
        loss = ctc_loss(probabilities, torch.tensor([], dtype=torch.long),
                        torch.tensor([frames]), torch.tensor([0]))
        assert abs(loss.item() + np.log(.8)) < 1e-5
    waveform = np.random.default_rng(12351).normal(0, .2, 16000).astype(np.float32)
    changed_future = waveform.copy()
    changed_future[8000:] *= 100
    original = prepare(waveform, frontend='causal')
    changed = prepare(changed_future, frontend='causal')
    assert np.array_equal(original[:16000], changed[:16000])
    assert np.isfinite(prepare(np.zeros(16000, np.float32), frontend='causal')).all()
    for index in range(6):
        legacy, legacy_targets = Data(6, 872119, [], frontend='legacy')[index]
        causal, causal_targets = Data(6, 872119, [], frontend='causal')[index]
        assert causal.shape == legacy.shape and causal.isfinite().all()
        assert torch.equal(legacy_targets, causal_targets)
    with patch.object(synth, 'case', wraps=synth.case) as generate:
        for index in range(12):
            audio, _ = Data(12, 872119, [], short_gap_fraction=1, element_gap_range=(.3, .3))[index]
            assert audio.shape == (224000,) and audio.isfinite().all()
        for call in generate.call_args_list:
            assert call.kwargs['element_gap_scale'] == (1 if call.args[1] == 'noise' else .3)
    print('CTC labels, blank normalization, causal frontend, unchanged targets and weighted-gap checks passed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--checkpoint', default='.research/cwformer-release/best_model.pt')
    parser.add_argument('--output', default='models/cwformer-experiment')
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--seed', type=int, default=67193811)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--frontend', choices=['legacy', 'causal'], default='legacy')
    parser.add_argument('--save-every', type=int, default=200)
    parser.add_argument('--short-gap-fraction', type=float, default=0)
    parser.add_argument('--element-gap-range', type=float, nargs=2, default=[.25, 1], metavar=('MIN', 'MAX'))
    parser.add_argument('--noise-recording', action='append', default=[])
    args = parser.parse_args()
    if args.check:
        self_check()
        return
    if args.steps < 1 or args.batch < 1 or args.workers < 0 or args.save_every < 1 or not 0 < args.lr < 1:
        parser.error('steps and batch must be positive; workers nonnegative; learning rate between zero and one')
    if not 0 <= args.short_gap_fraction <= 1 or not .1 <= args.element_gap_range[0] <= args.element_gap_range[1] <= 1:
        parser.error('short-gap fraction must be 0–1; element-gap range must satisfy 0.1 <= MIN <= MAX <= 1')
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    model, saved_config = load(args.checkpoint, args.device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        **vars(args), 'upstreamRevision': REVISION, 'torch': str(torch.__version__),
        'sourceSHA256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'frontendSHA256': hashlib.sha256(Path(sys.modules[Frontend.__module__].__file__).read_bytes()).hexdigest(),
        'generatorSHA256': hashlib.sha256(Path(synth.__file__).read_bytes()).hexdigest(),
        'initialCheckpointSHA256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()}
    (output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    batches = DataLoader(Data(args.steps * args.batch, args.seed, args.noise_recording, args.frontend,
                             args.short_gap_fraction, args.element_gap_range),
                         batch_size=args.batch, num_workers=args.workers, collate_fn=collate)
    started, running_loss = time.monotonic(), 0
    for step, (audio, targets, target_lengths) in enumerate(batches, 1):
        audio, targets, target_lengths = (value.to(args.device) for value in (audio, targets, target_lengths))
        optimizer.zero_grad(set_to_none=True)
        lengths = torch.full((len(audio),), audio.shape[1], device=args.device, dtype=torch.int64)
        probabilities, output_lengths = model(audio, lengths)
        loss = ctc_loss(probabilities, targets, output_lengths, target_lengths)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite CTC loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        running_loss += loss.item()
        if step % 20 == 0:
            print(json.dumps({'step': step, 'loss': running_loss / 20,
                              'seconds': time.monotonic() - started}), flush=True)
            running_loss = 0
        if step % args.save_every == 0 or step == args.steps:
            config = {**saved_config, 'max_cache_len': 250}
            torch.save({'model_state_dict': model.state_dict(), 'model_config': config,
                        'step': step, 'experiment': vars(args)}, output / f'step-{step}.pt')


if __name__ == '__main__':
    main()
