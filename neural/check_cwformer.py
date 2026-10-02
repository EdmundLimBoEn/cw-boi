"""Checks the exact streaming path, including arbitrary incoming block sizes."""
import argparse

import numpy as np
from morseformer.decoding.streaming import StreamingConfig
from morse_synth.operator import OperatorConfig, build_events
from morse_synth.keying import render_events

from cwformer_engine import Model, StreamingDecoder, decode_offline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    args = parser.parse_args()
    model = Model(args.model)
    cfg = StreamingConfig(carrier_hz=650, bandwidth_hz=150)
    signal = render_events(build_events('CQ TEST 73', OperatorConfig(wpm=20)), freq=650, sample_rate=8000, amplitude=.4, tail_ms=1000)
    expected = decode_offline(model, signal, cfg)
    assert expected == 'CQ TEST 73', repr(expected)
    leading_dash = render_events(build_events('TEST 73', OperatorConfig(wpm=20)), freq=650, sample_rate=8000, amplitude=.4, tail_ms=1000)
    assert decode_offline(model, leading_dash, cfg) == 'TEST 73'
    for size in (137, 4000, 8011):
        decoder = StreamingDecoder(model, cfg)
        result = ''.join(fragment for start in range(0, len(signal), size) for fragment in decoder.feed(signal[start:start + size])) + decoder.flush()
        assert result == expected, (size, result)
        assert decoder.flush() == ''
        try:
            decoder.feed(signal[:10])
            raise AssertionError('Finished decoder accepted audio')
        except ValueError:
            pass
    for value in (np.nan, np.inf):
        try:
            StreamingDecoder(model, cfg).feed(np.array([value], np.float32))
            raise AssertionError('Invalid sample accepted')
        except ValueError:
            pass
    decoder = StreamingDecoder(model, cfg)
    rng = np.random.default_rng(824511)
    text = ''
    for _ in range(120):
        text += ''.join(decoder.feed(rng.normal(0, .1, 4000).astype(np.float32)))
        assert len(decoder.pending) < 4000
        assert len(decoder.overlap) < 400
        assert all(value.shape[2] <= 250 for key, value in decoder.state.items() if key.startswith('kv_'))
    text += decoder.flush()
    assert not text, repr(text)
    print('CWformer live chunking, finalization, finite-input, noise and bounded-state checks passed.')


if __name__ == '__main__':
    main()
