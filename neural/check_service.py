"""Exercise a running local decoder service without opening an audio device."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, ProxyHandler, build_opener

import numpy as np
import torch
from morse_synth.keying import render_events
from morse_synth.operator import OperatorConfig, build_events
from morseformer.decoding.streaming import StreamingConfig

import listen


def run(url, engine, checkpoint):
    url = listen.local_url(url)
    opener = build_opener(ProxyHandler({}), listen.NoRedirect())
    sessions = set()

    def call(path, body=None, headers=None, status=200):
        request = Request(url + path, data=body, headers=headers or {}, method='GET' if body is None else 'POST')
        try:
            response = opener.open(request, timeout=30)
        except HTTPError as error:
            response = error
        with response:
            data = json.loads(response.read())
            expected = (status,) if isinstance(status, int) else status
            assert response.code in expected, (path, response.code, status, data)
            return data

    def start(frequency=711, bandwidth=150):
        key = call('/api/stream/start?' + urlencode({'frequency': frequency, 'bandwidth': bandwidth}), b'')['session']
        sessions.add(key)
        return key

    def action(name, key, body=b'', frequency=711, bandwidth=150, status=200):
        return call('/api/stream/' + name + '?' + urlencode({'session': key, 'frequency': frequency, 'bandwidth': bandwidth}), body, status=status)

    health = call('/api/health')
    assert health['ready'] and health['engine'] == engine
    assert health['checkpoint'] == f'{checkpoint.parent.name}/{checkpoint.name}', health
    torch.set_num_threads(4)
    if engine == 'cwformer':
        from cwformer_engine import Model, StreamingDecoder, decode_offline
        model = Model(checkpoint)
    else:
        from server import load_model
        from streaming import StreamingDecoder, decode_offline
        model = load_model(str(checkpoint))
    config = StreamingConfig(carrier_hz=711, bandwidth_hz=150,
                             confidence_threshold=health['confidence_threshold'] if engine == 'rnnt' else .6,
                             digit_threshold=health['digit_threshold'] if engine == 'rnnt' else .9)
    audio = render_events(build_events('CQ TEST 73', OperatorConfig(wpm=20)), freq=711,
                          sample_rate=8000, amplitude=.4, tail_ms=1000)
    raw = audio.astype('<f4').tobytes()
    with torch.inference_mode():
        direct = decode_offline(model, audio, config)
    assert direct == 'CQ TEST 73', (engine, 'direct engine', direct)
    try:
        offline = call('/api/decode?frequency=711&bandwidth=150', raw)['text']
        assert offline == direct, (engine, 'HTTP offline', offline, direct)
        key = start()
        for offset in range(0, len(audio), 8011):
            action('feed', key, audio[offset:offset + 8011].astype('<f4').tobytes())
        final = action('finish', key)['text']
        assert final == direct, (engine, 'HTTP stream', final, direct)
        action('finish', key, status=404)
        sessions.discard(key)

        key = start(1000, 300)
        retuned_cfg = StreamingConfig(carrier_hz=1000, bandwidth_hz=300, confidence_threshold=config.confidence_threshold,
                                      digit_threshold=config.digit_threshold)
        decoder = StreamingDecoder(model, retuned_cfg)
        quiet = np.zeros(4001, np.float32)
        with torch.inference_mode():
            wanted = ''.join(decoder.feed(quiet))
            action('feed', key, quiet.tobytes(), 1000, 300)
            retuned_cfg.carrier_hz, retuned_cfg.bandwidth_hz = 711, 80
            if engine == 'rnnt':
                decoder._fcfg.tone_freq, decoder._fcfg.bandwidth = 711, 80
            for offset in range(0, len(audio), 5037):
                block = audio[offset:offset + 5037]
                wanted += ''.join(decoder.feed(block))
                action('feed', key, block.astype('<f4').tobytes(), 711, 80)
            wanted += decoder.flush()
        actual = action('finish', key)['text']
        sessions.discard(key)
        assert actual == wanted.strip(), (engine, 'retune', actual, wanted)
        assert actual == 'CQ TEST 73', (engine, 'retune copy', actual)

        for headers in ({'Host': 'example.com'}, {'Host': '['}, {'Origin': 'https://evil.example'},
                        {'Origin': 'http://['}, {'Origin': 'ftp://localhost'}, {'Origin': 'null'}):
            call('/api/health', headers=headers, status=403)
            call('/api/stream/start', b'', headers=headers, status=403)
        call('/api/health', headers={'Origin': 'http://localhost:5173'})
        for query in ('frequency=nan', 'frequency=inf', 'frequency=249', 'frequency=1401',
                      'bandwidth=nan', 'bandwidth=39', 'bandwidth=501'):
            call('/api/stream/start?' + query, b'', status=400)
        key = start()
        for bad in (b'', b'abc', np.array([np.nan], '<f4').tobytes(), np.array([np.inf], '<f4').tobytes(),
                    np.array([101], '<f4').tobytes()):
            action('feed', key, bad, status=400)
            call('/api/decode', bad, status=400)
        action('cancel', key)
        action('feed', key, raw[:640], status=404)
        sessions.discard(key)
        call('/api/decode', b'', headers={'Content-Length': str(8000 * 600 * 4 + 4)}, status=413)
        for _ in range(4):
            start()
        call('/api/stream/start', b'', status=429)
        for key in list(sessions):
            action('cancel', key)
            sessions.discard(key)

        process = Mock()
        process.stdout.fileno.return_value = 123
        process.poll.return_value = None
        process.wait.return_value = 0
        source = BytesIO(raw)
        copy = StringIO()
        args = SimpleNamespace(url=url, input_index=1, frequency=711, bandwidth=150, seconds=60)
        with (patch.object(listen.subprocess, 'Popen', return_value=process),
              patch.object(listen.select, 'select', return_value=([process.stdout], [], [])),
              patch.object(listen.os, 'read', side_effect=lambda _fd, count: source.read(min(997, count))),
              redirect_stdout(copy), redirect_stderr(StringIO())):
            listen.listen(args)
        process.stdout.close.assert_called_once()
        assert copy.getvalue().strip() == direct, (engine, 'Terminal bridge', copy.getvalue(), direct)
        print(f'{engine}: HTTP/direct CQ TEST 73, retune, cancel, validation, origin checks, session limits and mocked capture passed.')
    finally:
        for key in sessions:
            action('cancel', key, status=(200, 404))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--engine', choices=['rnnt', 'cwformer'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    run(args.url, args.engine, args.checkpoint)
