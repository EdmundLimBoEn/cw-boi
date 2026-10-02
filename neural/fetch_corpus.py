"""Download the pinned public recordings used for CW experiments."""

import argparse
import hashlib
import json
import subprocess
import tempfile
import wave
from pathlib import Path
from urllib.request import Request, urlopen


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fetch(entry, root):
    target = root / entry['path']
    if target.is_file() and digest(target) == entry['sha256']:
        print(f"Verified {entry['path']}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.download')
    try:
        headers = {'Range': entry['range']} if 'range' in entry else {}
        with urlopen(Request(entry['url'], headers=headers), timeout=60) as response, temporary.open('wb') as out:
            received = 0
            while block := response.read(1024 * 1024):
                received += len(block)
                if received > entry['bytes']:
                    raise ValueError(f"Source size changed: {entry['path']}")
                out.write(block)
        if received != entry['bytes'] or digest(temporary) != entry['sha256']:
            raise ValueError(f"Source checksum changed: {entry['path']}")
        temporary.replace(target)
        print(f"Downloaded {entry['path']}")
    finally:
        temporary.unlink(missing_ok=True)


def prepare_holdout(project, root):
    clips = json.loads((project / 'neural/corpus-holdout.json').read_text())['clips']
    if all((root / clip['audio']).is_file() and digest(root / clip['audio']) == clip['sha256'] for clip in clips):
        return
    source = root / clips[0]['sourcePrefix']
    pcm = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(source),
                                   '-ar', '8000', '-ac', '1', '-f', 's16le', 'pipe:1'])
    for clip in clips:
        start, end = clip['sourceRangeSeconds']
        target = root / clip['audio']
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


def check():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / 'source'
        source.write_bytes(b'CW')
        entry = dict(path='copy', url=source.as_uri(), bytes=2, sha256=digest(source))
        fetch(entry, root)
        assert (root / 'copy').read_bytes() == b'CW'
        source.write_bytes(b'XX')
        (root / 'copy').write_bytes(b'previous')
        try:
            fetch(entry, root)
        except ValueError:
            pass
        else:
            raise AssertionError('A changed source was accepted')
        assert (root / 'copy').read_bytes() == b'previous'
        assert not (root / 'copy.download').exists()
    print('Corpus download integrity check passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', choices=['noise', 'marine-electric', 'saq', 'all'], default='all')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        check()
        raise SystemExit
    project = Path(__file__).resolve().parents[1]
    manifest = json.loads(Path(__file__).with_name('corpus.json').read_text())
    for entry in manifest['files']:
        group = 'noise' if entry['path'].startswith('cwformer/') else entry['path'].split('/')[0]
        if args.only in ('all', group):
            fetch(entry, project / manifest['download_root'])
    if args.only in ('all', 'marine-electric'):
        prepare_holdout(project, project / manifest['download_root'])
