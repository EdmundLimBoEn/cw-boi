"""Small, offline checks for the training generator and inference boundary."""

import numpy as np
from unittest.mock import patch

from server import samples_from_bytes, settings
from train import case, collate, continuous_waveform, WINDOW
from morse_synth.keying import render_events
from morse_synth.operator import build_events
from morseformer.core.tokenizer import encode, decode
import torch


def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Invalid input was accepted')


if __name__ == '__main__':
    for family in ('clean', 'rough', 'chaos', 'noise'):
        features, text = case(58219, family)
        repeated, label = case(58219, family)
        assert text == label and np.array_equal(features, repeated)
        assert features.shape == (3000, 1) and np.isfinite(features).all()
        assert decode(encode(text)) == text
        assert bool(text) == (family != 'noise')
        dense, dense_text = case(58319, family, continuous=True)
        assert dense.shape == (3000, 1) and np.isfinite(dense).all()
        assert decode(encode(dense_text)) == dense_text
    for seed in range(20):
        wave, text, spans = continuous_waveform(np.random.default_rng(seed), clean=False)
        assert len(wave) == WINDOW and np.isfinite(wave).all()
        assert text == ''.join(char for char, _, _ in spans).strip()
        assert all(0 <= start < end <= WINDOW for _, start, end in spans)
        assert any(end > WINDOW / 2 for _, _, end in spans), 'Continuous examples must include late characters.'
    with patch('train.render_events', wraps=render_events) as render, patch('train.build_events', wraps=build_events) as events:
        _, label = case(58219, 'clean', return_audio=True)
        normal = render.call_args.args[0]
        unit = 1.2 / events.call_args.args[1].wpm
        _, weighted_label = case(58219, 'clean', return_audio=True, element_gap_scale=.3)
        weighted = render.call_args.args[0]
        assert label == weighted_label and len(normal) == len(weighted)
        for index, ((on, duration), (weighted_on, weighted_duration)) in enumerate(zip(normal, weighted)):
            assert on == weighted_on
            ratio = .3 if index and not on and duration < 1.5 * unit else 1
            assert abs(weighted_duration - duration * ratio) < 1e-10, 'Only intra-character gaps may be scaled.'
    for family in ('clean', 'rough', 'chaos'):
        for seed in range(12):
            weighted, label = case(seed, family, return_audio=True, element_gap_scale=.25)
            assert len(weighted) == WINDOW and np.isfinite(weighted).all() and label
    for scale in (0, float('nan'), float('inf'), 1.1):
        rejected(lambda: case(1, 'clean', element_gap_scale=scale))
    batch, targets, lengths = collate([(torch.zeros(3000, 1), torch.tensor([], dtype=torch.long)),
                                      (torch.zeros(3000, 1), torch.tensor(encode('CQ')))])
    assert batch.shape == (2, 3000, 1) and lengths.tolist() == [0, 2]
    assert samples_from_bytes(np.zeros(160, dtype='<f4').tobytes()).shape == (160,)
    for body in (b'', b'abc', np.array([float('nan')], dtype='<f4').tobytes(), np.array([101], dtype='<f4').tobytes()):
        rejected(lambda: samples_from_bytes(body))
    for frequency in ('nan', 'inf', '0', '1500'):
        rejected(lambda: settings({'frequency': [frequency]}))
    rejected(lambda: settings({'bandwidth': ['-1']}))
    print('Generator reproducibility, complete labels, mixed silence batches, and input boundaries passed.')
