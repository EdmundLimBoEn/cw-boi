"""Replay scanner-selected carriers through CWformer, for files and live input."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from morseformer.decoding.streaming import StreamingConfig

from cwformer_engine import Model, StreamingDecoder, decode_offline


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def errors(expected, actual):
    previous = list(range(len(actual) + 1))
    for i, left in enumerate(expected, 1):
        row = [i]
        for j, right in enumerate(actual, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = row
    return previous[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traces', required=True)
    parser.add_argument('--model')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    trace_path = Path(args.traces)
    trace_set = json.loads(trace_path.read_text())
    manifest_path = trace_path.parent / trace_set['manifest']
    if digest(manifest_path) != trace_set['manifestSha256']:
        raise ValueError('Audio manifest differs from the scanner replay')
    manifest = json.loads(manifest_path.read_text())
    if manifest['sampleRate'] != 8000 or manifest['format'] != 'float32-le-mono':
        raise ValueError('Expected 8 kHz float32 mono fixtures')
    traces = {item['id']: item for item in trace_set['traces']}
    model = Model(args.model)
    results = []
    for item in manifest['cases']:
        audio_path = manifest_path.parent / item['file']
        if digest(audio_path) != item['sha256']:
            raise ValueError(f"{item['id']}: audio changed after acquisition")
        audio = np.fromfile(audio_path, dtype='<f4')
        trace = traces[item['id']]
        cfg = StreamingConfig(carrier_hz=650, bandwidth_hz=150)
        decoder = StreamingDecoder(model, cfg)
        live, start = '', 0
        for chunk in trace['chunks']:
            stop = chunk['stop']
            if not start < stop <= len(audio):
                raise ValueError(f"{item['id']}: invalid streaming sample boundary")
            cfg.carrier_hz = chunk['frequency']
            live += ''.join(decoder.feed(audio[start:stop]))
            start = stop
        if start != len(audio):
            raise ValueError(f"{item['id']}: missing streaming audio")
        live = (live + decoder.flush()).strip()
        file_text = None if item.get('liveOnly') else decode_offline(model, audio, StreamingConfig(carrier_hz=trace['fileFrequency'], bandwidth_hz=150))
        result = {key: item[key] for key in ('id', 'condition', 'expected', 'frequency', 'durationSeconds')}
        result.update(referenceCharacters=len(item['expected']), live=live, liveErrors=errors(item['expected'], live),
                      file=file_text, fileErrors=errors(item['expected'], file_text) if file_text is not None else None,
                      fileReferenceCharacters=len(item['expected']) if file_text is not None else 0,
                      liveFalseCharacters=sum(not char.isspace() for char in live) if not item['expected'] else None,
                      fileFalseCharacters=sum(not char.isspace() for char in file_text) if not item['expected'] and file_text is not None else None,
                      finalFrequency=trace['finalFrequency'], acquired=abs(trace['finalFrequency'] - item['frequency']) <= 20,
                      fileFrequency=trace['fileFrequency'], fileAcquired=abs(trace['fileFrequency'] - item['frequency']) <= 20 if file_text is not None else None)
        if item['condition'] == 'crowded':
            before = [entry for entry in trace['history'] if entry['time'] <= item['interfererOnset']]
            after = [entry for entry in trace['history'] if entry['time'] > item['interfererOnset']]
            checked = before[-1:] + after
            result['offTargetUpdates'] = sum(abs(entry['frequency'] - item['frequency']) > 20 for entry in checked)
        if item['condition'] == 'new-carrier':
            times = [entry['time'] - item['onset'] for entry in trace['history']
                     if entry['time'] >= item['onset'] and abs(entry['frequency'] - item['frequency']) <= 20]
            result['reacquisitionSeconds'] = min(times) if times else None
        results.append(result)
        print(json.dumps({'id': item['id'], 'liveErrors': result['liveErrors'], 'fileErrors': result['fileErrors']}), flush=True)
    summary = {}
    for condition in dict.fromkeys(item['condition'] for item in results):
        subset = [item for item in results if item['condition'] == condition]
        files = [item for item in subset if item['file'] is not None]
        summary[condition] = {'cases': len(subset), 'referenceCharacters': sum(item['referenceCharacters'] for item in subset),
                              'liveErrors': sum(item['liveErrors'] for item in subset),
                              'fileCases': len(files), 'fileReferenceCharacters': sum(item['fileReferenceCharacters'] for item in files),
                              'fileErrors': sum(item['fileErrors'] for item in files) if files else None,
                              'liveFalseCharacters': sum(item['liveFalseCharacters'] for item in subset) if condition == 'noise' else None,
                              'fileFalseCharacters': sum(item['fileFalseCharacters'] for item in files) if condition == 'noise' and files else None,
                              'acquired': sum(item['acquired'] for item in subset),
                              'fileAcquired': sum(item['fileAcquired'] for item in files) if files else None,
                              'durationSeconds': sum(item['durationSeconds'] for item in subset)}
    model_files = [model.path, model.path.parent / 'mel_config.json', model.path.parent / 'mel_basis.npy', model.path.parent / 'mel_window.npy']
    report = {'version': 1, 'seed': manifest['seed'], 'protocol': trace_set['protocol'],
              'manifestSha256': digest(manifest_path), 'traceSha256': digest(trace_path), 'scannerPipeline': trace_set['pipeline'],
              'model': [{'file': str(path), 'sha256': digest(path)} for path in model_files],
              'runtime': [{'file': str(path), 'sha256': digest(path)} for path in [Path(__file__), Path(__file__).with_name('cwformer_engine.py')]],
              'summary': summary, 'cases': results}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
