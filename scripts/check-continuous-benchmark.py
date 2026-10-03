"""Check seed validation and byte-identical compatibility with the first release."""

import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

BASELINE = '87ced16fb4402c2836d02aeff2c8a434afa58a81'
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = 'scripts/continuous-benchmark.ts'


def run(script, output, *args):
    return subprocess.run(['bun', str(script), '--output', str(output), *args],
                          cwd=ROOT, capture_output=True, text=True)


with TemporaryDirectory(prefix='cw-generator-check-') as directory:
    temporary = Path(directory)
    for file in [SCRIPT, 'src/signals.ts', 'src/morse.ts']:
        target = temporary / 'baseline' / file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.check_output(['git', 'show', f'{BASELINE}:{file}'], cwd=ROOT))
    for split, seed in [('dev', 817351), ('final', 917351)]:
        manifests = []
        for name, script, arguments in [
            ('old', temporary / 'baseline' / SCRIPT, []),
            ('default', ROOT / SCRIPT, []),
            ('explicit', ROOT / SCRIPT, ['--seed', str(seed)]),
        ]:
            output = temporary / f'{split}-{name}'
            result = run(script, output, '--split', split, *arguments)
            assert result.returncode == 0, result.stderr
            manifest = json.loads((output / 'manifest.json').read_text())
            assert manifest['seed'] == seed and len(manifest['cases']) == 24
            manifests.append((output, manifest))
        for output, manifest in manifests[1:]:
            assert manifest['cases'] == manifests[0][1]['cases']
            for case in manifest['cases']:
                assert (output / case['file']).read_bytes() == (manifests[0][0] / case['file']).read_bytes()
    for seed in ['', '0', '-1', '1.5', 'NaN', 'Infinity', '4294967296', ' 1', '1e3']:
        result = run(ROOT / SCRIPT, temporary / 'invalid', f'--seed={seed}')
        assert result.returncode != 0 and '--seed must be an integer' in result.stderr
        assert not (temporary / 'invalid').exists()
    result = run(ROOT / SCRIPT, temporary / 'maximum', '--seed', '4294967295', '--cases-per-condition', '1')
    assert result.returncode == 0, result.stderr
    assert json.loads((temporary / 'maximum' / 'manifest.json').read_text())['seed'] == 4294967295

print('Default dev/final audio and case records are byte-identical; explicit seeds and invalid/uint32 boundary checks pass.')
