"""Check file and live acquisition gates separately on paired scanner replays."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def compare(baseline, candidate, final=False):
    assert baseline['manifestSha256'] == candidate['manifestSha256'], 'Different acquisition corpora'
    assert [item['sha256'] for item in baseline['model']] == [item['sha256'] for item in candidate['model']], 'Different models/frontends'
    assert [item['sha256'] for item in baseline['runtime']] == [item['sha256'] for item in candidate['runtime']], 'Different neural runtimes'
    before = {item['id']: item for item in baseline['cases']}
    after = {item['id']: item for item in candidate['cases']}
    assert len(before) == len(baseline['cases']) and len(after) == len(candidate['cases'])
    assert before.keys() == after.keys()
    for key, left in before.items():
        right = after[key]
        assert left['expected'] == right['expected'] and left['condition'] == right['condition']
        assert left['durationSeconds'] == right['durationSeconds']
        assert (left['file'] is None) == (right['file'] is None)
    gates, paths = {}, {}
    for mode in ['live', 'file']:
        speech = [key for key, item in before.items() if item['expected'] and item[mode] is not None]
        clean = [key for key in speech if before[key]['condition'] == 'normal-clean']
        noise = [key for key, item in before.items() if not item['expected'] and item[mode] is not None]
        assert speech and clean and noise
        characters = sum(len(before[key]['expected']) for key in speech)
        old_errors = sum(before[key][f'{mode}Errors'] for key in speech)
        new_errors = sum(after[key][f'{mode}Errors'] for key in speech)
        clean_characters = sum(len(before[key]['expected']) for key in clean)
        clean_delta = sum(after[key][f'{mode}Errors'] - before[key][f'{mode}Errors'] for key in clean) / clean_characters
        old_noise = sum(before[key][f'{mode}FalseCharacters'] for key in noise)
        new_noise = sum(after[key][f'{mode}FalseCharacters'] for key in noise)
        low = [item for item in after.values() if item['condition'] == 'low-amplitude']
        acquired = 'acquired' if mode == 'live' else 'fileAcquired'
        fraction = sum(item[acquired] for item in low) / len(low)
        crowded = [item for item in after.values() if item['condition'] == 'crowded']
        assert low and crowded
        gates.update({f'{mode}RelativeImprovementAtLeast10Percent': old_errors > 0 and (old_errors - new_errors) / old_errors >= .10,
                      f'{mode}AbsoluteImprovementAtLeastOnePoint': (old_errors - new_errors) / characters >= .01,
                      f'{mode}NormalCleanNoninferiorHalfPoint': clean_delta <= .005,
                      f'{mode}NoiseFalseCharactersNonincrease': new_noise <= old_noise,
                      f'{mode}LowAmplitudeWithin20HzAtLeast95Percent': fraction >= .95,
                      f'{mode}CrowdedKeepsDesiredStation': all(item[acquired] for item in crowded)})
        paths[mode] = {'speechCases': len(speech), 'characters': characters, 'baselineErrors': old_errors,
                       'candidateErrors': new_errors, 'baselineCER': old_errors / characters, 'candidateCER': new_errors / characters,
                       'normalCleanDeltaCER': clean_delta, 'noiseCases': len(noise),
                       'noiseSeconds': sum(before[key]['durationSeconds'] for key in noise),
                       'baselineNoiseFalseCharacters': old_noise, 'candidateNoiseFalseCharacters': new_noise,
                       'lowAmplitudeAcquiredFraction': fraction}
        if final:
            gates[f'{mode}EightMinutesOfFreshFinalNoise'] = paths[mode]['noiseSeconds'] == 480 and baseline['seed'] == candidate['seed'] == 4401901
    crowded = [item for item in after.values() if item['condition'] == 'crowded']
    new_station = [item for item in after.values() if item['condition'] == 'new-carrier']
    assert new_station
    gates['liveCrowdedNoOffTargetUpdates'] = all(item['offTargetUpdates'] == 0 for item in crowded)
    gates['newStationAcquiredWithinOneSecond'] = all(item['reacquisitionSeconds'] is not None and item['reacquisitionSeconds'] <= 1 for item in new_station)
    return {'allGuardsPass': all(gates.values()), 'gates': gates, 'paths': paths,
            'newStationReacquisitionSeconds': [item['reacquisitionSeconds'] for item in new_station],
            'claimLimits': 'Fixed message templates with fresh waveform parameters. Gaussian-only negatives; no real-operator claim. Actual queue replay excludes browser resampling, physical audio capture, and HTTP latency.'}


def main():
    if sys.argv[1:] == ['--self-check']:
        cases = []
        for condition in ['low-amplitude', 'normal-clean', 'crowded', 'new-carrier', 'noise']:
            expected = '' if condition == 'noise' else 'TEST TEST'
            cases.append({'id': condition, 'condition': condition, 'expected': expected, 'durationSeconds': 20,
                          'live': '', 'file': None if condition == 'new-carrier' else '', 'liveErrors': 0,
                          'fileErrors': None if condition == 'new-carrier' else 0,
                          'liveFalseCharacters': 0, 'fileFalseCharacters': 0, 'acquired': True,
                          'fileAcquired': True, 'offTargetUpdates': 0, 'reacquisitionSeconds': .5})
        candidate = {'manifestSha256': 'same', 'model': [], 'runtime': [], 'seed': 3401901, 'cases': cases}
        baseline = json.loads(json.dumps(candidate))
        for case in baseline['cases']:
            if case['condition'] in ['low-amplitude', 'crowded']:
                case['liveErrors'] = case['fileErrors'] = 5
        result = compare(baseline, candidate)
        assert result['allGuardsPass'] and result['paths']['file']['characters'] == 27
        assert result['paths']['live']['characters'] == 36
        candidate['cases'][0]['fileErrors'] = 10
        assert not compare(baseline, candidate)['allGuardsPass'], 'Live improvement masked file failure'
        print('Separate file/live gates and live-only denominator check: PASS')
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    reports = [json.loads(Path(path).read_text()) for path in [args.baseline, args.candidate]]
    report = compare(*reports, final=args.final)
    report['sources'] = [{'file': path, 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()} for path in [args.baseline, args.candidate]]
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'allGuardsPass': report['allGuardsPass'], 'gates': report['gates'], 'paths': report['paths']}))


if __name__ == '__main__':
    main()
