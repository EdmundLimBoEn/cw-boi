"""Evaluate complete recordings through the same decoder used by the app."""

import argparse
import hashlib
import inspect
from importlib.metadata import version
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import server


def errors(expected, actual):
    previous = list(range(len(actual) + 1))
    for i, left in enumerate(expected, 1):
        row = [i]
        for j, right in enumerate(actual, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = row
    return previous[-1]


def aggregate(rows):
    speech = [row for row in rows if row['expected']]
    noise = [row for row in rows if not row['expected']]
    characters = sum(len(row['expected']) for row in speech)
    total_errors = sum(row['errors'] for row in speech)
    noise_seconds = sum(row['durationSeconds'] for row in noise)
    false_chars = sum(len(row['actual'].replace(' ', '')) for row in noise)
    return {'cases': len(rows), 'speechCases': len(speech), 'characters': characters,
            'errors': total_errors, 'cer': total_errors / characters if characters else None,
            'exact': sum(row['expected'] == row['actual'] for row in speech),
            'noiseCases': len(noise), 'noiseSeconds': noise_seconds, 'falseCharacters': false_chars,
            'falseCharactersPerMinute': false_chars * 60 / noise_seconds if noise_seconds else None,
            'audioSeconds': sum(row['durationSeconds'] for row in rows),
            'inferenceSeconds': sum(row['inferenceSeconds'] for row in rows)}


def self_check():
    assert errors('CQ DE K1ABC', 'CQDE K1ABC') == 1
    assert errors('AAA', 'A') == 2 and errors('', 'CQ') == 2
    result = aggregate([
        {'expected': 'CQ', 'actual': 'C', 'errors': 1, 'durationSeconds': 10, 'inferenceSeconds': 1},
        {'expected': '', 'actual': 'E T', 'errors': 3, 'durationSeconds': 20, 'inferenceSeconds': 1}])
    assert result['cer'] == 0.5 and result['falseCharactersPerMinute'] == 6
    assert result['exact'] == 0 and result['noiseCases'] == 1
    from tempfile import TemporaryDirectory
    from scipy.io import wavfile
    with TemporaryDirectory() as directory:
        root = Path(directory)
        audio_path = root / 'stereo.wav'
        wavfile.write(audio_path, 16000, np.full((16000, 2), 16384, dtype=np.int16))
        source = root / 'source.json'
        source.write_text(json.dumps({'clips': [{'audio': str(audio_path), 'expected': 'TEST',
                                                'startSeconds': 0.25, 'endSeconds': 0.75,
                                                'frequency': 650, 'operator': 'conversion-check'}]}))
        target = root / 'converted' / 'manifest.json'
        prepare(source, target)
        case = json.loads(target.read_text())['cases'][0]
        converted = np.fromfile(target.parent / case['file'], dtype='<f4')
        assert len(converted) == 4000 and np.allclose(converted[20:-20], 0.5, atol=0.001)
        assert case['expected'] == 'TEST' and case['provenance']['operator'] == 'conversion-check'
    print('Edit distance, noise rates, WAV scaling, cropping, stereo conversion and resampling checks passed.')



def prepare(source_path, output_path):
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    source_path, output_path = Path(source_path), Path(output_path)
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    rows = source if isinstance(source, list) else source['clips']
    if not rows:
        raise ValueError('No reviewed clips available in the source manifest')
    base = Path('.') if isinstance(source, list) else Path(source.get('download_root', '.'))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    loaded, cases, identifiers = {}, [], set()
    for index, row in enumerate(rows):
        audio_path = base / row.get('audio', row.get('path', ''))
        if audio_path not in loaded:
            if audio_path.suffix.lower() == '.f32':
                audio = np.fromfile(audio_path, dtype='<f4')
                rate = int(row.get('sampleRate', 8000))
            else:
                rate, audio = wavfile.read(audio_path)
                dtype = audio.dtype
                audio = audio.astype(np.float32)
                if np.issubdtype(dtype, np.unsignedinteger):
                    midpoint = (np.iinfo(dtype).max + 1) / 2
                    audio = (audio - midpoint) / midpoint
                elif np.issubdtype(dtype, np.signedinteger):
                    audio /= float(2 ** (np.iinfo(dtype).bits - 1))
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
            if audio.ndim != 1 or rate <= 0 or not np.isfinite(audio).all():
                raise ValueError(f'Invalid source audio: {audio_path}')
            loaded[audio_path] = (rate, audio, hashlib.sha256(audio_path.read_bytes()).hexdigest())
        rate, audio, source_sha = loaded[audio_path]
        start = float(row.get('startSeconds', 0))
        end = float(row.get('endSeconds', len(audio) / rate))
        if not math.isfinite(start + end) or not 0 <= start < end <= len(audio) / rate + 1 / rate:
            raise ValueError(f'Invalid crop interval: {audio_path} [{start}, {end}]')
        crop = audio[round(start * rate):round(end * rate)]
        if rate != 8000:
            divisor = math.gcd(rate, 8000)
            crop = resample_poly(crop, 8000 // divisor, rate // divisor)
        raw = crop.astype('<f4').tobytes()
        server.samples_from_bytes(raw)
        identifier = str(row.get('id', f"{row.get('condition', row.get('challenge', 'real'))}-{index + 1}"))
        if identifier in identifiers:
            raise ValueError(f'Duplicate case id: {identifier}')
        identifiers.add(identifier)
        file = f'clip-{index + 1:03d}.f32'
        (output_path.parent / file).write_bytes(raw)
        expected = row['expected'] if 'expected' in row else row['text']
        cases.append({'id': identifier, 'condition': row.get('condition', row.get('challenge', 'real')),
                      'expected': expected, 'file': file, 'sha256': hashlib.sha256(raw).hexdigest(),
                      'sampleRate': 8000, 'samples': len(crop), 'durationSeconds': len(crop) / 8000,
                      'frequency': row['frequency'], 'bandwidth': row.get('bandwidth', 100),
                      'sourceSHA256': source_sha, 'provenance': row})
    manifest = {'version': 1, 'sampleRate': 8000, 'format': 'float32-le-mono', 'knownCarrier': True,
                'split': 'supplementary-dev' if isinstance(source, list) else source.get('split', 'real-dev'),
                'sourceManifest': str(source_path), 'sourceManifestSHA256': hashlib.sha256(source_bytes).hexdigest(),
                'cases': cases}
    output_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'manifest': str(output_path), 'cases': len(cases),
                      'durationSeconds': sum(case['durationSeconds'] for case in cases)}))


def run(args):
    if args.self_check:
        self_check()
        return
    if args.prepare:
        if not args.output:
            raise ValueError('--output manifest path is required with --prepare')
        prepare(args.prepare, args.output)
        return
    if not args.manifest:
        raise ValueError('--manifest is required')
    manifest_path = Path(args.manifest)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get('format') != 'float32-le-mono' or manifest.get('sampleRate') != 8000:
        raise ValueError('Manifest must contain 8 kHz little-endian float32 mono audio')
    cases = manifest['cases']
    if args.conditions:
        cases = [case for case in cases if case['condition'] in args.conditions.split(',')]
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        raise ValueError('No benchmark cases selected')
    checkpoint = Path(args.checkpoint or server.hf_hub_download('sderhy/morseformer', server.MODEL, revision=server.REVISION))
    torch.set_num_threads(args.threads)
    server.device = args.device
    model = server.load_model(str(checkpoint))
    decoder_class, offline_decoder = server.StreamingDecoder, server.decode_offline
    if getattr(args, 'decoder', 'app') == 'original':
        from morseformer.decoding.streaming import StreamingDecoder, decode_offline
        decoder_class, offline_decoder = StreamingDecoder, decode_offline
    results = []
    report = {'version': 1, 'split': manifest.get('split'), 'seed': manifest.get('seed'),
              'manifest': str(manifest_path), 'manifestSHA256': hashlib.sha256(manifest_bytes).hexdigest(),
              'checkpoint': str(checkpoint), 'checkpointSHA256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'mode': args.mode, 'decoder': getattr(args, 'decoder', 'app'), 'chunkSeconds': args.chunk_seconds, 'device': args.device,
              'torchVersion': str(torch.__version__), 'morseformerVersion': version('morseformer'),
              'decoderSHA256': hashlib.sha256(Path(inspect.getfile(decoder_class)).read_bytes()).hexdigest(),
              'serverSHA256': hashlib.sha256(Path(server.__file__).read_bytes()).hexdigest(),
              'benchmarkSHA256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'knownCarrier': manifest.get('knownCarrier'),
              'complete': False, 'results': results}
    for case in cases:
        raw = (manifest_path.parent / case['file']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != case['sha256']:
            raise ValueError(f"Audio hash mismatch: {case['id']}")
        audio = server.samples_from_bytes(raw)
        if len(audio) != case['samples']:
            raise ValueError(f"Wrong audio length: {case['id']}")
        config = server.settings({'frequency': [str(case['frequency'])], 'bandwidth': [str(case.get('bandwidth', 100))]})
        config.confidence_threshold = getattr(args, 'confidence_threshold', config.confidence_threshold)
        config.digit_threshold = getattr(args, 'digit_threshold', config.digit_threshold)
        started = time.perf_counter()
        fragments = []
        with torch.inference_mode():
            if args.mode == 'offline':
                actual = offline_decoder(model, audio, config, args.device)
            else:
                decoder = decoder_class(model, config, args.device)
                chunk = max(1, round(args.chunk_seconds * config.sample_rate))
                for start in range(0, len(audio), chunk):
                    for fragment in decoder.feed(audio[start:start + chunk]):
                        fragments.append({'atSeconds': min(start + chunk, len(audio)) / config.sample_rate, 'text': fragment})
                tail = decoder.flush()
                if tail:
                    fragments.append({'atSeconds': len(audio) / config.sample_rate, 'text': tail, 'final': True})
                actual = ''.join(fragment['text'] for fragment in fragments)
        actual = actual.strip().upper()
        expected = case['expected'].strip().upper()
        distance = errors(expected, actual)
        row = {'id': case['id'], 'condition': case['condition'], 'expected': expected, 'actual': actual,
               'errors': distance, 'cer': distance / len(expected) if expected else None,
               'errorsWithoutSpaces': errors(expected.replace(' ', ''), actual.replace(' ', '')),
               'durationSeconds': len(audio) / 8000, 'inferenceSeconds': time.perf_counter() - started,
               'settings': asdict(config), 'fragments': fragments}
        results.append(row)
        print(f"{len(results)}/{len(cases)} {case['id']}: {distance} errors | {actual}", file=sys.stderr, flush=True)
    report['complete'] = True
    report['aggregate'] = aggregate(results)
    report['conditions'] = {condition: aggregate([row for row in results if row['condition'] == condition])
                            for condition in sorted({row['condition'] for row in results})}
    rendered = json.dumps(report, indent=2) + '\n'
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding='utf-8')
        print(json.dumps({'output': str(target), 'aggregate': report['aggregate'], 'conditions': report['conditions']}))
    else:
        print(rendered)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest')
    parser.add_argument('--prepare', help='Convert reviewed corpus clips or streaming sweep JSON into a raw-audio manifest')
    parser.add_argument('--checkpoint')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--mode', choices=['stream', 'offline'], default='stream')
    parser.add_argument('--decoder', choices=['app', 'original'], default='app', help='Use original published streaming algorithm for a controlled baseline')
    parser.add_argument('--chunk-seconds', type=float, default=1.0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--confidence-threshold', type=float, default=0.6)
    parser.add_argument('--digit-threshold', type=float, default=0.9)
    parser.add_argument('--conditions', help='Comma-separated condition filter for development diagnostics')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if not np.isfinite(args.chunk_seconds) or args.chunk_seconds <= 0 or args.threads < 1:
        parser.error('chunk-seconds and threads must be positive')
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (args.confidence_threshold, args.digit_threshold)):
        parser.error('Confidence and digit thresholds must be finite probabilities in [0, 1]')
    if args.limit is not None and args.limit < 1:
        parser.error('limit must be positive')
    run(args)
