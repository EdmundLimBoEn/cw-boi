"""Package the released fp32 model as byte-identical Cloudflare-size assets."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def prepare(source: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    model = source / 'cwformer_streaming_fp32.onnx'
    digest = hashlib.sha256()
    parts = []
    with model.open('rb') as stream:
        while data := stream.read(20 * 1024 * 1024):
            digest.update(data)
            checksum = hashlib.sha256(data).hexdigest()
            filename = f'model-{len(parts)}-{checksum[:16]}.bin'
            (destination / filename).write_bytes(data)
            parts.append({'file': filename, 'bytes': len(data), 'sha256': checksum})
    if digest.hexdigest() != '8344b39dba22e595ea8170bd4ffdb514aaef1c4b67b7de3ef965b5f66a081374':
        raise ValueError('Expected the verified CWformer v6 step-2000 fp32 release')
    assets = {}
    for name, shape in [('window', (400,)), ('basis', (40, 201))]:
        data = np.load(source / f'mel_{name}.npy')
        if data.shape != shape or not np.isfinite(data).all():
            raise ValueError(f'Invalid {name} asset')
        raw = np.asarray(data, dtype='<f4').tobytes()
        filename = f'mel_{name}.f32'
        (destination / filename).write_bytes(raw)
        assets[name] = {'file': filename, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    license_path = source / 'LICENSE'
    if not license_path.exists():
        license_path = Path(__file__).resolve().parents[1] / 'neural/CWFORMER_LICENSE'
    (destination / 'LICENSE').write_bytes(license_path.read_bytes())
    manifest = {'format': 'cwformer-v6', 'modelBytes': sum(part['bytes'] for part in parts),
                'modelSha256': digest.hexdigest(), 'parts': parts,
                **assets}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'destination': str(destination), 'bytes': manifest['modelBytes'], 'parts': len(parts)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('models/cwformer-weighted-v6/step-2000'))
    parser.add_argument('--destination', type=Path, default=Path('public/models/cwformer-v6'))
    args = parser.parse_args()
    prepare(args.source, args.destination)
