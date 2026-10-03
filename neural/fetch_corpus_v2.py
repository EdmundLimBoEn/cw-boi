"""Fetch a versioned real-audio corpus and reproduce its pinned WAV crops."""
import argparse
import json
import subprocess
import tempfile
import wave
from pathlib import Path

from fetch_corpus import digest, fetch


def prepare(manifest, project):
    root = project / manifest['download_root']
    for entry in manifest.get('files', []):
        fetch(entry, root)
    pending = [c for c in manifest['clips'] if not (root / c['audio']).is_file()
               or digest(root / c['audio']) != c['sha256']]
    for source in sorted({c['sourceAudio'] for c in pending}):
        clips = [c for c in pending if c['sourceAudio'] == source]
        end = max(c['sourceRangeSeconds'][1] for c in clips)
        pcm = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(root / source),
                                       '-t', str(end + 1), '-ar', '8000', '-ac', '1',
                                       '-f', 's16le', 'pipe:1'])
        for clip in clips:
            start, end = clip['sourceRangeSeconds']
            if not 0 <= start < end <= len(pcm) / 16000:
                raise ValueError(f"Invalid or incomplete crop: {clip['id']}")
            target = root / clip['audio']
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.derived.wav')
            try:
                with wave.open(str(temporary), 'wb') as out:
                    out.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
                    out.writeframes(pcm[round(start * 8000) * 2:round(end * 8000) * 2])
                if digest(temporary) != clip['sha256']:
                    raise ValueError(f"Converted waveform changed: {clip['id']}")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            print(f"Derived {clip['id']}")


def check():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        with wave.open(str(root / 'source.wav'), 'wb') as out:
            out.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
            out.writeframes(b'\x00\x01' * 16000)
        with wave.open(str(root / 'expected.wav'), 'wb') as out:
            out.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
            out.writeframes(b'\x00\x01' * 8000)
        clip = dict(id='check', sourceAudio='source.wav', audio='out.wav',
                    sourceRangeSeconds=[.25, 1.25], sha256=digest(root / 'expected.wav'))
        manifest = dict(download_root='.', clips=[clip])
        prepare(manifest, root)
        assert digest(root / 'out.wav') == clip['sha256']
        clip['sha256'] = '0' * 64
        previous = (root / 'out.wav').read_bytes()
        try:
            prepare(manifest, root)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed conversion accepted')
        assert (root / 'out.wav').read_bytes() == previous
        assert not (root / 'out.derived.wav').exists()
    print('Versioned corpus crop integrity check passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('neural/corpus-dev-v2.json'))
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        check()
    else:
        prepare(json.loads(args.manifest.read_text()), Path(__file__).resolve().parents[1])
