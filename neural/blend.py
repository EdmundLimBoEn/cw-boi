"""Interpolate effective checkpoint weights, including the published EMA overlay."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

import server


def blend(base, candidate, fraction):
    if not 0 <= fraction <= 1 or base.keys() != candidate.keys():
        raise ValueError('Fraction must be in [0, 1] and checkpoint keys must match')
    mixed = {}
    for key, value in base.items():
        other = candidate[key]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError(f'Incompatible checkpoint tensor: {key}')
        if value.is_floating_point():
            mixed[key] = torch.lerp(value, other, fraction)
        elif torch.equal(value, other):
            mixed[key] = value.clone()
        else:
            raise ValueError(f'Cannot interpolate different integer buffers: {key}')
    return mixed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate')
    parser.add_argument('--base')
    parser.add_argument('--fractions', type=float, nargs='+', default=[0.25, 0.5, 0.75])
    parser.add_argument('--output', default='models/continuous-gpu-v3-blends')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        left, right = {'x': torch.tensor([2., 4.])}, {'x': torch.tensor([6., 12.])}
        assert torch.equal(blend(left, right, 0)['x'], left['x'])
        assert torch.equal(blend(left, right, 1)['x'], right['x'])
        assert torch.equal(blend(left, right, 0.25)['x'], torch.tensor([3., 6.]))
        print('Weight interpolation and endpoint checks passed.')
        raise SystemExit(0)
    if not args.candidate:
        parser.error('--candidate is required')
    base_path = Path(args.base or server.hf_hub_download('sderhy/morseformer', server.MODEL, revision=server.REVISION))
    candidate_path = Path(args.candidate)
    base = server.load_model(str(base_path)).state_dict()
    candidate = server.load_model(str(candidate_path)).state_dict()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for fraction in args.fractions:
        provenance = {'candidateFraction': fraction, 'base': str(base_path), 'candidate': str(candidate_path),
                      'baseSHA256': hashlib.sha256(base_path.read_bytes()).hexdigest(),
                      'candidateSHA256': hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                      'effectiveEMA': True}
        path = output / f'candidate-{fraction:g}.pt'
        torch.save({'model': blend(base, candidate, fraction), 'blend': provenance}, path)
        provenance['checkpointSHA256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_suffix('.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'checkpoint': str(path), **provenance}))
