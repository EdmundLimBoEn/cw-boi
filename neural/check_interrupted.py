"""Check interrupted fixtures and stream-timing scoring without changing runtime copy."""

import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from morseformer.decoding.streaming import StreamingConfig

import benchmark_cwformer as benchmark
import cwformer_engine


class EmittingDecoder:
    def __init__(self, *_):
        pass

    def feed(self, _):
        return [' X']

    def flush(self):
        return ' END'


config = StreamingConfig(carrier_hz=650, bandwidth_hz=150)
parameters = {'pauseScoringGraceSeconds': 2, 'bursts': [
    {'startSample': 0, 'lastMarkSample': 8000},
    {'startSample': 48000, 'lastMarkSample': 56000},
]}
with patch.object(benchmark.cwformer_engine, 'StreamingDecoder', EmittingDecoder):
    text, timing = benchmark.decode_with_pause_timing(None, np.zeros(112100, np.float32), config, parameters)
    assert timing['pauseFalseCharacters'] == 16 and timing['pauseScoredSeconds'] == 8
    assert text.endswith(' END') and not timing['emissions'][-1]['insideScoredPause']
    invalid = {**parameters, 'bursts': [{'startSample': 2, 'lastMarkSample': 1}]}
    try:
        benchmark.decode_with_pause_timing(None, np.zeros(112000, np.float32), config, invalid)
        raise AssertionError('Invalid interval accepted')
    except ValueError:
        pass

root = Path(__file__).resolve().parents[1]
model = cwformer_engine.Model()
with TemporaryDirectory(prefix='cw-interrupted-check-') as temporary:
    temporary = Path(temporary)
    manifests = []
    for repeat in range(2):
        output = temporary / str(repeat)
        result = subprocess.run(['bun', 'scripts/interrupted-benchmark.ts', '--seed', '7104127',
                                 '--cases-per-condition', '1', '--output', str(output)],
                                cwd=root, check=True, capture_output=True, text=True)
        manifests.append(json.loads((output / 'manifest.json').read_text()))
    assert manifests[0] == manifests[1]
    assert len(manifests[0]['cases']) == 3
    for case in manifests[0]['cases']:
        data = (temporary / '0' / case['file']).read_bytes()
        assert data == (temporary / '1' / case['file']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == case['sha256']
        audio = np.frombuffer(data, dtype='<f4')
        assert len(audio) == case['samples'] and np.isfinite(audio).all()
        bursts = case['parameters']['bursts']
        assert case['expected'] == ' '.join(burst['expected'] for burst in bursts)
        assert len(bursts) == 3 and len(audio) - bursts[-1]['lastMarkSample'] >= 4 * 8000
        assert np.std(audio[bursts[-1]['lastMarkSample'] + 8000:]) > .001
        config = StreamingConfig(carrier_hz=case['frequency'], bandwidth_hz=150)
        original = cwformer_engine.decode_offline(model, audio, config)
        timed, timing = benchmark.decode_with_pause_timing(model, audio, config, case['parameters'])
        assert timed == original, (case['id'], original, timed)
        assert timing['pauseScoredSeconds'] > 0
print('Interrupted corpus reproducibility, complete labels, continuing receiver noise, pause scoring, and unchanged streaming copy: PASS')
