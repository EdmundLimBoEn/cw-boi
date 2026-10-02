"""Score recordings through the deployed causal CWformer runtime."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
from morseformer.decoding.streaming import StreamingConfig

from benchmark import aggregate, errors
import cwformer_engine


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--model', help='FP32 ONNX model; defaults to the selected app checkpoint')
    parser.add_argument('--bandwidth', type=float, default=150)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('format') != 'float32-le-mono' or manifest.get('sampleRate') != 8000:
        raise ValueError('Manifest must contain 8 kHz little-endian float32 mono audio')
    if not 20 <= args.bandwidth <= 1200:
        raise ValueError('Bandwidth must be between 20 and 1200 Hz')
    model = cwformer_engine.Model(args.model)
    results = []
    for case in manifest['cases']:
        path = manifest_path.parent / case['file']
        if digest(path) != case['sha256']:
            raise ValueError(f'Audio hash mismatch: {path}')
        audio = np.fromfile(path, dtype='<f4')
        if len(audio) != case['samples']:
            raise ValueError(f'Audio sample count mismatch: {path}')
        config = StreamingConfig(carrier_hz=case['frequency'], bandwidth_hz=args.bandwidth)
        start = time.perf_counter()
        actual = cwformer_engine.decode_offline(model, audio, config)
        elapsed = time.perf_counter() - start
        row = {'id': case['id'], 'condition': case['condition'], 'expected': case['expected'],
               'actual': actual, 'errors': errors(case['expected'], actual),
               'durationSeconds': len(audio) / 8000, 'inferenceSeconds': elapsed}
        print(f"{row['id']}: {row['errors']} errors, {actual!r}", flush=True)
        results.append(row)
    report = {'version': 1, 'split': manifest.get('split'), 'manifest': str(manifest_path),
              'manifestSHA256': digest(manifest_path), 'model': str(model.path),
              'modelSHA256': digest(model.path), 'decoderSHA256': digest(cwformer_engine.__file__),
              'benchmarkSHA256': digest(__file__), 'onnxruntimeVersion': version('onnxruntime'),
              'provider': 'CPUExecutionProvider', 'knownCarrier': manifest.get('knownCarrier'),
              'bandwidth': args.bandwidth, 'complete': True, 'results': results,
              'aggregate': aggregate(results),
              'conditions': {condition: aggregate([row for row in results if row['condition'] == condition])
                             for condition in sorted({row['condition'] for row in results})}}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['aggregate']), flush=True)


if __name__ == '__main__':
    main()
