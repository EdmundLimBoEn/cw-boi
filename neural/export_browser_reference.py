"""Export bounded, consumed-fixture references for the visitor-device port.

Run on the evaluation host, not on the thermally constrained development Mac.
This compares implementations of the released model; it is not a new accuracy
benchmark and intentionally never opens the reserved final corpus.
"""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '2'

import numpy as np
import onnxruntime as ort

from cwformer_engine import Frontend, Model, StreamingDecoder


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode(model, audio, frequency, chunk_size=4000, retune=None):
    settings = SimpleNamespace(carrier_hz=frequency, bandwidth_hz=150)
    decoder = StreamingDecoder(model, settings)
    fragments = []
    for start in range(0, len(audio), chunk_size):
        if retune and start >= retune['sample']:
            settings.carrier_hz = retune['frequency']
        fragments.append(''.join(decoder.feed(audio[start:start + chunk_size])))
    tail = decoder.flush()
    assert decoder.flush() == ''
    return {'fragments': fragments, 'tail': tail, 'text': ''.join(fragments) + tail}


def export_frontend(model, audio, out):
    cases = []
    for bandwidth in (80, 100, 150):
        frontend = Frontend(650, bandwidth)
        overlap = np.zeros(200, np.float32)

        def mel(samples):
            nonlocal overlap
            joined = np.concatenate((overlap, samples))
            count = max(0, (len(joined) - 400) // 160 + 1)
            overlap = joined[count * 160:].copy()
            frames = np.lib.stride_tricks.sliding_window_view(joined, 400)[::160][:count]
            power = np.abs(np.fft.rfft(frames * model.window, axis=1)) ** 2
            return np.log(power @ model.basis.T + 1e-6).astype(np.float32)

        mel(np.zeros(16000, np.float32))
        blocks = []
        for index in range(3):
            frontend.frequency = 650 if index == 0 else 657
            samples = frontend.process(audio[index * 4000:(index + 1) * 4000])
            features = mel(samples)
            sample_file = f'frontend-{bandwidth}-{index}.f32'
            feature_file = f'mel-{bandwidth}-{index}.f32'
            samples.astype('<f4').tofile(out / sample_file)
            features.astype('<f4').tofile(out / feature_file)
            blocks.append({'frequency': frontend.frequency, 'audio': sample_file,
                           'mel': feature_file, 'frames': len(features)})
        cases.append({'bandwidth': bandwidth, 'blocks': blocks})
    return {'inputFile': 'clean-1.f32', 'blockSamples': 4000, 'warmupSamples': 16000, 'cases': cases}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    original_session = ort.InferenceSession

    def limited_session(path, options=None, **kwargs):
        options = options or ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        return original_session(path, options, **kwargs)

    ort.InferenceSession = limited_session
    model = Model(args.model)
    reference = {
        'version': 1,
        'purpose': 'Implementation parity on previously consumed development fixtures; not a fresh accuracy claim.',
        'modelSHA256': sha256(args.model),
        'pythonRuntimeSHA256': sha256(Path(__file__).with_name('cwformer_engine.py')),
        'exporterSHA256': sha256(__file__),
        'providers': model.session.get_providers(),
        'onnxruntime': ort.__version__,
        'threads': {'intraOp': 2, 'interOp': 1},
        'bandwidth': 150,
        'cases': [],
    }
    selections = [
        ('round3-dev', 'clean-1'),
        ('round3-dev', 'human-1'),
        ('round3-dev', 'rough-1'),
        ('round3-dev', 'chaos-1'),
        ('round3-dev', 'noise-1-1'),
        ('round3-dev', 'noise-3-1'),
        ('round3-interrupted-dev', 'clean-paused-1'),
    ]
    started = time.monotonic()
    for dataset, case_id in selections:
        manifest_path = args.data_root / dataset / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        case = next(item for item in manifest['cases'] if item['id'] == case_id)
        source = manifest_path.parent / case['file']
        assert sha256(source) == case['sha256'], f'Waveform changed: {source}'
        audio = np.fromfile(source, dtype='<f4')
        assert len(audio) == case['samples'] and np.isfinite(audio).all()
        shutil.copyfile(source, args.out / f'{case_id}.f32')
        result = {
            'id': case_id,
            'dataset': dataset,
            'condition': case['condition'],
            'noiseKind': case.get('parameters', {}).get('kind'),
            'sourceManifestSHA256': sha256(manifest_path),
            'audioSHA256': case['sha256'],
            'file': f'{case_id}.f32',
            'samples': len(audio),
            'frequency': case['frequency'],
            'expected': case['expected'],
            'reference': decode(model, audio, case['frequency']),
        }
        if case_id == 'clean-1':
            reference['frontend'] = export_frontend(model, audio, args.out)
            result['chunking'] = {str(size): decode(model, audio, case['frequency'], size)
                                  for size in (137, 8011, len(audio))}
            assert all(item['text'] == result['reference']['text'] for item in result['chunking'].values())
            # A retune with 1,370 samples queued checks that the old passband's
            # incomplete chunk is discarded instead of decoded at the new tone.
            result['retune'] = {'sample': 1370, 'frequency': 650, 'chunkSize': 137}
            result['retune']['reference'] = decode(model, audio, case['frequency'], 137, result['retune'])
        reference['cases'].append(result)
        print(f"{case_id}: {result['reference']['text']!r}", flush=True)
    reference['elapsedSeconds'] = time.monotonic() - started
    (args.out / 'reference.json').write_text(json.dumps(reference, indent=2) + '\n')
    print(f"Wrote {len(reference['cases'])} references in {reference['elapsedSeconds']:.1f}s", flush=True)


if __name__ == '__main__':
    main()
