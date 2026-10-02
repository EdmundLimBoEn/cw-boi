"""Mix synthetic CW holdouts with reserved receiver noise; this is not human CW."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.io import wavfile

REFERENCE_RMS = 0.4 / math.sqrt(2)
TAG = 'mixed-real-receiver-noise'
PROJECT = Path(__file__).resolve().parents[1]
SNR_DEFINITION = ('Nominal keyed carrier RMS (0.4/sqrt(2)) / added recorded-noise RMS at 8 kHz, '
                  'before receiver filtering. Existing synthetic noise and fading remain unchanged; this is not total SNR.')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_noise(path):
    rate, audio = wavfile.read(path)
    if rate != 8000 or audio.ndim != 1 or not len(audio):
        raise ValueError('Receiver noise must be a nonempty mono 8 kHz WAV.')
    dtype = audio.dtype
    audio = audio.astype(np.float32)
    if np.issubdtype(dtype, np.unsignedinteger):
        midpoint = (np.iinfo(dtype).max + 1) / 2
        audio = (audio - midpoint) / midpoint
    elif np.issubdtype(dtype, np.signedinteger):
        audio /= float(2 ** (np.iinfo(dtype).bits - 1))
    if not np.isfinite(audio).all():
        raise ValueError('Receiver noise contains nonfinite samples.')
    return audio


def mix(manifest_path, noise_path, output, seed=8321907, per_condition=6, corpus_path=None):
    manifest_path, noise_path, output = map(Path, (manifest_path, noise_path, output))
    if not 1 <= per_condition <= 6 or seed < 0:
        raise ValueError('Use 1–6 cases per condition and a nonnegative seed.')
    source_bytes = manifest_path.read_bytes()
    source = json.loads(source_bytes)
    if source.get('sampleRate') != 8000 or source.get('format') != 'float32-le-mono':
        raise ValueError('Input manifest must describe 8 kHz little-endian float32 mono audio.')
    corpus = json.loads(Path(corpus_path or Path(__file__).with_name('corpus.json')).read_text())
    noise_sha = sha256(noise_path.read_bytes())
    provenance = next((entry for entry in corpus['files'] if entry['sha256'] == noise_sha
                       and entry.get('kind') == 'real_receiver_noise' and entry.get('split') == 'holdout'), None)
    if provenance is None:
        raise ValueError('Noise is not a checksum-matched receiver holdout in corpus.json.')
    noise = read_noise(noise_path)
    selected = []
    for condition in ('clean', 'rough'):
        rows = [row for row in source['cases'] if row['condition'] == condition and row['expected'].strip()][:per_condition]
        if not rows:
            raise ValueError(f'No labelled {condition} cases in the source manifest.')
        for row in rows:
            raw = (manifest_path.parent / row['file']).read_bytes()
            if sha256(raw) != row['sha256'] or not raw or len(raw) % 4:
                raise ValueError(f"Invalid source audio or checksum: {row['id']}")
            audio = np.frombuffer(raw, dtype='<f4')
            if len(audio) != row['samples'] or not np.isfinite(audio).all() or len(audio) > len(noise):
                raise ValueError(f"Invalid source length/samples, or noise recording too short: {row['id']}")
            selected.append((row, audio))
    files = [f'mixed-{index + 1:03d}.f32' for index in range(len(selected) * 2)]
    if output.exists() or any((output.parent / file).exists() for file in files):
        raise ValueError('Output manifest or audio already exists; choose a fresh output directory.')
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cases = []
    for row, audio in selected:
        offset = int(rng.integers(0, len(noise) - len(audio) + 1))
        segment = noise[offset:offset + len(audio)]
        rms = float(np.sqrt(np.mean(segment.astype(np.float64) ** 2)))
        if rms <= 1e-12:
            raise ValueError('Selected recorded-noise segment has zero RMS.')
        for snr in (10, 0):
            gain = REFERENCE_RMS / (10 ** (snr / 20) * rms)
            mixed = (audio + segment * gain).astype('<f4')
            if not np.isfinite(mixed).all() or np.max(np.abs(mixed)) > 100:
                raise ValueError('Mixed audio exceeds the decoder input range; no clipping was applied.')
            raw = mixed.tobytes()
            file = files[len(cases)]
            with (output.parent / file).open('xb') as stream:
                stream.write(raw)
            cases.append({**row, 'id': f"{row['id']}-receiver-{snr}db", 'condition': f"{TAG}-{row['condition']}-{snr}db",
                          'file': file, 'sha256': sha256(raw), 'tag': TAG, 'sourceCase': row,
                          'receiverNoise': {'sha256': noise_sha, 'sourceOffsetSamples': offset,
                                            'sourceOffsetSeconds': offset / 8000, 'gain': gain,
                                            'sourceRMS': rms, 'addedRMS': rms * gain,
                                            'additionalNoiseSNRDb': snr, 'keyedCarrierReferenceRMS': REFERENCE_RMS}})
    result = {'version': 1, 'split': source.get('split'), 'tag': TAG, 'seed': seed,
              'sampleRate': 8000, 'format': 'float32-le-mono', 'knownCarrier': source.get('knownCarrier', True),
              'snrDefinition': SNR_DEFINITION, 'sourceManifest': str(manifest_path.resolve()),
              'sourceManifestSHA256': sha256(source_bytes),
              'provenance': {'kind': TAG, 'description': 'Synthetic CW plus additive recorded receiver noise; not real human-sent CW.',
                             'sourceManifestMetadata': {key: value for key, value in source.items() if key != 'cases'},
                             'noiseFile': str(noise_path.resolve()), 'noiseSHA256': noise_sha, 'noiseSource': provenance,
                             'mixerSHA256': sha256(Path(__file__).read_bytes())}, 'cases': cases}
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    return result


def self_check():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        noise_path = root / 'noise.wav'
        wavfile.write(noise_path, 8000, np.tile(np.array([0.25, -0.25], np.float32), 12000))
        assert np.array_equal(read_noise(noise_path)[:2], [0.25, -0.25])
        integer_path = root / 'integer.wav'
        wavfile.write(integer_path, 8000, np.array([-16384, 16384], np.int16))
        assert np.array_equal(read_noise(integer_path), [-0.5, 0.5])
        raw = np.linspace(-0.4, 0.4, 8000, dtype='<f4').tobytes()
        (root / 'cw.f32').write_bytes(raw)
        source = {'sampleRate': 8000, 'format': 'float32-le-mono', 'split': 'self-check',
                  'cases': [{'id': condition, 'condition': condition, 'expected': 'CQ TEST 73', 'frequency': 711,
                             'samples': 8000, 'durationSeconds': 1, 'file': 'cw.f32', 'sha256': sha256(raw),
                             'parameters': {'snr': 19}} for condition in ('clean', 'rough')]}
        manifest_path = root / 'source.json'
        manifest_path.write_text(json.dumps(source))
        corpus_path = root / 'corpus.json'
        corpus_path.write_text(json.dumps({'files': [{'sha256': sha256(noise_path.read_bytes()),
                                                    'kind': 'real_receiver_noise', 'split': 'holdout', 'notes': 'Synthetic test fixture.'}]}))
        outputs = [root / name / 'manifest.json' for name in ('one', 'two')]
        for output in outputs:
            result = mix(manifest_path, noise_path, output, per_condition=1, corpus_path=corpus_path)
            assert len(result['cases']) == 4 and result['tag'] == TAG
            for case in result['cases']:
                assert case['expected'] == 'CQ TEST 73' and case['frequency'] == 711 and case['samples'] == 8000
                assert case['parameters']['snr'] == 19 and case['sourceCase']['sha256'] == sha256(raw)
                mixed = np.fromfile(output.parent / case['file'], dtype='<f4')
                added = mixed - np.frombuffer(raw, dtype='<f4')
                wanted = REFERENCE_RMS / 10 ** (case['receiverNoise']['additionalNoiseSNRDb'] / 20)
                assert math.isclose(float(np.sqrt(np.mean(added.astype(np.float64) ** 2))), wanted, rel_tol=1e-6)
        assert outputs[0].read_bytes() == outputs[1].read_bytes()
        try:
            mix(manifest_path, noise_path, outputs[0], per_condition=1, corpus_path=corpus_path)
        except ValueError:
            pass
        else:
            raise AssertionError('Existing holdout was overwritten.')
    print('Float/integer WAV scaling, deterministic mixing, preserved labels, added-noise RMS and overwrite checks passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, help='Existing synthetic benchmark manifest.')
    parser.add_argument('--noise', type=Path, default=PROJECT / '.research/data/cwformer/noise_40m_day.wav')
    parser.add_argument('--output', type=Path, help='New manifest path; accompanying audio goes into the same directory.')
    parser.add_argument('--seed', type=int, default=8321907)
    parser.add_argument('--cases-per-condition', type=int, default=6, help='First 1–6 labelled clean and rough cases, each at 10 and 0 dB.')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif not args.manifest or not args.output:
        parser.error('--manifest and --output are required.')
    else:
        result = mix(args.manifest, args.noise, args.output, args.seed, args.cases_per_condition)
        print(json.dumps({'manifest': str(args.output), 'tag': TAG, 'cases': len(result['cases']), 'seed': args.seed}))
