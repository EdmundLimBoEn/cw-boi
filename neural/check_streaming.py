"""Run with .venv/bin/python neural/check_streaming.py; no model download needed."""

import numpy as np
from morseformer.core.tokenizer import encode
from morseformer.decoding.streaming import StreamingConfig
from streaming import StreamingDecoder


def event(text, positions):
    return list(zip(encode(text), positions))


if __name__ == '__main__':
    decoder = StreamingDecoder(None, StreamingConfig())
    # The space falls just after the old cutoff, just before the new one.
    assert decoder._commit_tokens([(19, 74000), (1, 80000)], 64000, 80000) == 'R'
    assert decoder._commit_tokens([(19, 74000), (1, 79936), (19, 80800)], 80000, 96000) == ' R'
    # A previously emitted token crossing in the other direction stays emitted once.
    decoder = StreamingDecoder(None, StreamingConfig())
    assert decoder._commit_tokens(event('CQ', [16000, 31936]), 0, 32000) == 'CQ'
    assert decoder._commit_tokens(event('QDE', [32064, 36000, 40000]), 32000, 48000) == 'DE'
    # Closely repeated letters keep their separate acoustic timestamps.
    decoder = StreamingDecoder(None, StreamingConfig())
    assert decoder._commit_tokens(event('EEE', [31488, 32128, 32768]), 0, 32000) == 'E'
    assert decoder._commit_tokens(event('EEE', [31552, 32064, 32704]), 32000, 48000) == 'EE'
    # Unmatched left-context tokens are never invented into the transcript.
    decoder = StreamingDecoder(None, StreamingConfig())
    assert decoder._commit_tokens(event('CQ', [1000, 9000]), 0, 32000) == 'CQ'
    assert decoder._commit_tokens(event('XDE', [31936, 36000, 40000]), 32000, 48000) == 'DE'
    assert len(decoder._boundary) == 0
    decoder = StreamingDecoder(None, StreamingConfig())
    assert decoder.feed(np.zeros(80000, dtype=np.float32)) == []
    assert decoder._total_samples == 0 and decoder.flush() == ''
    tiny = np.array([1e-12, 0, 0, 0], dtype=np.float32)
    assert decoder.feed(np.concatenate([np.zeros(16000, dtype=np.float32), tiny])) == []
    assert np.array_equal(decoder._buffer, np.concatenate([np.zeros(1600, dtype=np.float32), tiny]))
    decoder.feed(np.zeros(800, dtype=np.float32))
    assert decoder._total_samples == 2404 and len(decoder._buffer) == 2404
    # Bounded pre-roll gives the same buffer irrespective of input chunk boundaries.
    other = StreamingDecoder(None, StreamingConfig())
    other.feed(np.concatenate([np.zeros(24000, dtype=np.float32), tiny, np.zeros(800, dtype=np.float32)]))
    assert np.array_equal(other._buffer, decoder._buffer)
    short = StreamingDecoder(None, StreamingConfig())
    original = np.concatenate([np.zeros(400, dtype=np.float32), tiny])
    short.feed(original)
    assert np.array_equal(short._buffer, original)
    print('Window-boundary copy, bounded pre-roll, weak onsets, internal silence, and chunk invariance passed.')
