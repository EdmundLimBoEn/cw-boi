"""Apply the frozen round-three gates to paired per-clip reports."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys


def paired(baseline, candidate, conditions=None):
    assert baseline['manifestSHA256'] == candidate['manifestSHA256'], 'Different corpora'
    original = {row['id']: row for row in baseline['results']}
    changed = {row['id']: row for row in candidate['results']}
    assert len(original) == len(baseline['results']) and len(changed) == len(candidate['results'])
    assert original.keys() == changed.keys(), 'Different cases'
    rows = []
    for key, left in original.items():
        right = changed[key]
        assert left['expected'] == right['expected'] and left['condition'] == right['condition']
        if left['expected'] and (conditions is None or left['condition'] in conditions):
            rows.append((left['condition'], len(left['expected']), left['errors'], right['errors']))
    assert rows, 'No speech cases'
    chars = sum(row[1] for row in rows)
    old, new = sum(row[2] for row in rows), sum(row[3] for row in rows)
    strata = defaultdict(list)
    for row in rows:
        strata[row[0]].append(row)
    rng, deltas = random.Random(7341), []
    for _ in range(10000):
        sample = [rng.choice(group) for group in strata.values() for _ in group]
        deltas.append(sum(row[3] - row[2] for row in sample) / sum(row[1] for row in sample))
    deltas.sort()
    return {'cases': len(rows), 'characters': chars, 'baselineErrors': old, 'candidateErrors': new,
            'baselineCER': old / chars, 'candidateCER': new / chars, 'deltaCER': (new - old) / chars,
            'absoluteReduction': (old - new) / chars, 'relativeReduction': (old - new) / old if old else 0,
            'pairedDeltaCER95PercentileInterval': [deltas[249], deltas[9749]]}


def compare(standard_base, standard_new, interrupted_base, interrupted_new, lane):
    assert standard_base['aggregate']['noiseSeconds'] == standard_new['aggregate']['noiseSeconds']
    assert interrupted_base['pauseTiming']['scoredSeconds'] == interrupted_new['pauseTiming']['scoredSeconds']
    standard, interrupted = paired(standard_base, standard_new), paired(interrupted_base, interrupted_new)
    target = interrupted if lane == 'receiver' else paired(standard_base, standard_new, {'rough', 'chaos'})
    gates = {
        'targetRelativeImprovementAtLeast10Percent': target['relativeReduction'] >= .10,
        'targetAbsoluteImprovementAtLeastOnePoint': target['absoluteReduction'] >= .01,
        'standardAggregateNoninferiorHalfPoint': standard['deltaCER'] <= .005,
        'negativeFalseCharactersNonincrease': standard_new['aggregate']['falseCharacters'] <= standard_base['aggregate']['falseCharacters'],
        'pauseFalseCharactersNonincrease': interrupted_new['pauseTiming']['falseCharacters'] <= interrupted_base['pauseTiming']['falseCharacters'],
    }
    conditions = {}
    for condition in ['clean', 'human', 'fading', 'crowded', 'rough', 'chaos']:
        conditions[condition] = paired(standard_base, standard_new, {condition})
        gates[f'{condition}Noninferior'] = conditions[condition]['deltaCER'] <= (.005 if condition in ['clean', 'human'] else .02)
    if lane == 'hard-noise-model':
        gates['interruptedNoninferiorHalfPoint'] = interrupted['deltaCER'] <= .005
    return {'lane': lane, 'allSyntheticGuardsPass': all(gates.values()), 'gates': gates,
            'target': target, 'standard': standard, 'interrupted': interrupted, 'conditions': conditions,
            'pauseFalseCharacters': {'baseline': interrupted_base['pauseTiming']['falseCharacters'], 'candidate': interrupted_new['pauseTiming']['falseCharacters']},
            'limit': 'Synthetic gates only. Apply any reserved real-session guard separately; do not treat correlated real crops as independent operators.'}


def main():
    if sys.argv[1:] == ['--self-check']:
        before = {'manifestSHA256': 'x', 'results': [{'id': 'one', 'condition': 'rough', 'expected': 'ABCD', 'errors': 2}]}
        after = {'manifestSHA256': 'x', 'results': [{'id': 'one', 'condition': 'rough', 'expected': 'ABCD', 'errors': 1}]}
        result = paired(before, after)
        assert result['deltaCER'] == -.25 and result['relativeReduction'] == .5
        assert result['pairedDeltaCER95PercentileInterval'] == [-.25, -.25]
        try:
            paired(before, {**after, 'manifestSHA256': 'different'})
            raise ValueError('Different corpora accepted')
        except AssertionError as error:
            assert str(error) == 'Different corpora'
        print('Paired comparison and bootstrap check: PASS')
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane', choices=['receiver', 'hard-noise-model'], required=True)
    for name in ['baseline-standard', 'candidate-standard', 'baseline-interrupted', 'candidate-interrupted', 'output']:
        parser.add_argument(f'--{name}', required=True)
    args = parser.parse_args()
    paths = [args.baseline_standard, args.candidate_standard, args.baseline_interrupted, args.candidate_interrupted]
    reports = [json.loads(Path(path).read_text()) for path in paths]
    report = compare(*reports, args.lane)
    report['sources'] = [{'file': path, 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()} for path in paths]
    report['uncertainty'] = '10000 paired whole-clip bootstrap resamples stratified by condition; seed 7341; descriptive 95% interval. Two predeclared lanes are tested, with no post-final candidate selection.'
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'lane': args.lane, 'pass': report['allSyntheticGuardsPass'], 'gates': report['gates']}))


if __name__ == '__main__':
    main()
