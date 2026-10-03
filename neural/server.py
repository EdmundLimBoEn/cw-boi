"""Local-only acoustic inference. Audio is kept in memory, never saved or logged."""

import argparse
import hashlib
import json
import math
import sys
import threading
import time
import uuid
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from morseformer.decoding.streaming import StreamingConfig
from streaming import StreamingDecoder, decode_offline
from morseformer.models.rnnt import RnntModel
from scripts.decode_audio import _rnnt_cfg_from_state

MODEL = 'rnnt_phase11b.pt'
engine_name = 'rnnt'
confidence_threshold = 0.6
digit_threshold = 0.9
RELEASE_CHECKPOINT = Path(__file__).resolve().parents[1] / 'models/cw-boi-rnnt-v1.pt'
REVISION = '9eab86a3ad7482f8c5801eabf26ec46f9493b919'
MAX_SAMPLES = 8000 * 600
device = 'cpu'
engines = {}
sessions = {}
# ponytail: one local model lock; use per-device workers if multiple users need inference.
model_lock = threading.Lock()


def load_model(path=None):
    checkpoint = path or hf_hub_download('sderhy/morseformer', MODEL, revision=REVISION)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = dict(saved['model'])
    state.update({key: value for key, value in saved.get('ema', {}).items() if key in state})
    network = RnntModel(_rnnt_cfg_from_state(state))
    network.load_state_dict(state)
    return network.to(device).eval()


def settings(query, engine=None):
    engine = engine or {}
    carrier = float(query.get('frequency', ['650'])[0])
    bandwidth = float(query.get('bandwidth', [engine.get('bandwidth', 100)])[0])
    if not math.isfinite(carrier) or not 250 <= carrier <= 1400:
        raise ValueError('Carrier must be 250–1400 Hz.')
    if not math.isfinite(bandwidth) or not 40 <= bandwidth <= 500:
        raise ValueError('Bandwidth must be 40–500 Hz.')
    return StreamingConfig(carrier_hz=carrier, bandwidth_hz=bandwidth,
                           confidence_threshold=engine.get('confidence_threshold', confidence_threshold),
                           digit_threshold=engine.get('digit_threshold', digit_threshold))


def requested_engine(query, default=None):
    values = query.get('engine', [default or engine_name])
    if len(values) != 1 or values[0] not in engines:
        raise ValueError('Choose one available decoder engine: ' + ', '.join(engines) + '.')
    return values[0]


def engine_info(name):
    engine = engines[name]
    return {key: engine[key] for key in ('model', 'checkpoint', 'device', 'bandwidth',
                                        'confidence_threshold', 'digit_threshold')} | {
        'engine': name, 'base_revision': REVISION if name == 'rnnt' else None}


def load_rnnt(checkpoint=None, published=False, confidence=None, digit=None):
    name = 'RNN-T · custom checkpoint' if checkpoint else 'Morseformer 0.6.4'
    acoustic, numeric = .6, .9
    if not checkpoint and not published and RELEASE_CHECKPOINT.exists():
        checkpoint = str(RELEASE_CHECKPOINT)
        release = json.loads(Path(__file__).with_name('release.json').read_text())
        if hashlib.sha256(RELEASE_CHECKPOINT.read_bytes()).hexdigest() != release['sha256']:
            raise ValueError('Local release checkpoint hash differs from neural/release.json; use --checkpoint for an explicit experiment.')
        name = release['name']
        acoustic, numeric = release['confidence_threshold'], release['digit_threshold']
    acoustic = acoustic if confidence is None else confidence
    numeric = numeric if digit is None else digit
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (acoustic, numeric)):
        raise ValueError('Acoustic confidence thresholds must be between 0 and 1.')
    path = Path(checkpoint) if checkpoint else None
    return {'model': name, 'network': load_model(checkpoint), 'stream': StreamingDecoder, 'decode': decode_offline,
            'checkpoint': f'{path.parent.name}/{path.name}' if path else MODEL, 'device': device, 'bandwidth': 100,
            'confidence_threshold': acoustic, 'digit_threshold': numeric}


def load_cwformer(checkpoint=None):
    from cwformer_engine import Model as CwformerModel, StreamingDecoder as CwformerStream, decode_offline as cwformer_decode
    from onnxruntime.capi.onnxruntime_pybind11_state import Fail, InvalidArgument, InvalidGraph, InvalidProtobuf, NoSuchFile, NotImplemented, RuntimeException
    try:
        network = CwformerModel(checkpoint)
    except (Fail, InvalidArgument, InvalidGraph, InvalidProtobuf, NoSuchFile, NotImplemented, RuntimeException) as error:
        raise RuntimeError(f'CWformer model could not load: {error}') from error
    return {'model': 'CWformer · v6' if checkpoint is None else 'CWformer · custom', 'network': network, 'stream': CwformerStream, 'decode': cwformer_decode,
            'checkpoint': f'{network.path.parent.name}/{network.path.name}', 'device': 'cpu', 'bandwidth': 150,
            'confidence_threshold': None, 'digit_threshold': None}


def samples_from_bytes(body):
    if not body or len(body) % 4 or len(body) > MAX_SAMPLES * 4:
        raise ValueError('Supply 8 kHz mono float32 audio, up to 10 minutes.')
    audio = np.frombuffer(body, dtype='<f4').copy()
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) > 100:
        raise ValueError('Audio contains invalid sample values.')
    return audio


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def respond(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def local_request(self):
        try:
            local = {'localhost', '127.0.0.1', '::1'}
            host = urlsplit('http://' + self.headers.get('Host', '')).hostname
            origin = self.headers.get('Origin')
            parsed = urlsplit(origin) if origin else None
            return host in local and (parsed is None or parsed.scheme in {'http', 'https'} and parsed.hostname in local)
        except ValueError:
            return False

    def do_GET(self):
        if not self.local_request():
            self.respond(403, {'error': 'This engine accepts local requests only.'})
        elif self.path == '/api/health':
            self.respond(200, {'ready': True, **engine_info(engine_name),
                               'engines': [engine_info(name) for name in engines]})
        else:
            self.respond(404, {'error': 'Unknown endpoint.'})

    def do_POST(self):
        if not self.local_request():
            return self.respond(403, {'error': 'This engine accepts local requests only.'})
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query, keep_blank_values=True)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 <= length <= MAX_SAMPLES * 4:
                return self.respond(413, {'error': 'Audio is limited to 10 minutes.'})
            self.connection.settimeout(30)
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError('Incomplete audio request.')
            with model_lock, torch.inference_mode():
                now = time.monotonic()
                for key in list(sessions):
                    if now - sessions[key]['touched'] > 120:
                        del sessions[key]
                if parsed.path == '/api/decode':
                    name = requested_engine(query)
                    engine = engines[name]
                    audio = samples_from_bytes(body)
                    config = settings(query, engine)
                    if len(audio) < 160:
                        text = ''
                    else:
                        text = engine['decode'](engine['network'], audio, config, engine['device'])
                    return self.respond(200, {'text': text.strip(), 'model': engine['checkpoint'], 'engine': name})
                if parsed.path == '/api/stream/start':
                    name = requested_engine(query)
                    engine = engines[name]
                    if len(sessions) >= 4:
                        return self.respond(429, {'error': 'Four decoder sessions are already active. Stop one and retry.'})
                    key = uuid.uuid4().hex
                    sessions[key] = {'engine': name, 'decoder': engine['stream'](engine['network'], settings(query, engine), engine['device']),
                                     'text': '', 'touched': now, 'samples': 0}
                    return self.respond(200, {'session': key, 'engine': name})
                key = query.get('session', [''])[0]
                session = sessions.get(key)
                if not session:
                    return self.respond(404, {'error': 'Decoder session expired. Start listening again.'})
                if requested_engine(query, session['engine']) != session['engine']:
                    raise ValueError('A decoder session cannot change engines. Start a new session.')
                engine = engines[session['engine']]
                session['touched'] = now
                if parsed.path == '/api/stream/feed':
                    tuned = settings(query, engine)
                    audio = samples_from_bytes(body)
                    if session['samples'] + len(audio) > MAX_SAMPLES:
                        del sessions[key]
                        return self.respond(413, {'error': 'Neural sessions are limited to 10 minutes. Start a new session.'})
                    session['decoder'].cfg.carrier_hz = tuned.carrier_hz
                    session['decoder'].cfg.bandwidth_hz = tuned.bandwidth_hz
                    if session['engine'] == 'rnnt':
                        session['decoder']._fcfg.tone_freq = tuned.carrier_hz
                        session['decoder']._fcfg.bandwidth = tuned.bandwidth_hz
                    session['samples'] += len(audio)
                    session['text'] += ''.join(session['decoder'].feed(audio))
                elif parsed.path == '/api/stream/finish':
                    if session['samples'] >= 160:
                        session['text'] += session['decoder'].flush()
                    del sessions[key]
                elif parsed.path == '/api/stream/cancel':
                    del sessions[key]
                else:
                    return self.respond(404, {'error': 'Unknown endpoint.'})
                self.respond(200, {'text': session['text'].strip(), 'engine': session['engine']})
        except (ValueError, TypeError, TimeoutError) as error:
            self.respond(400, {'error': str(error)})
        except Exception as error:
            print(f'Inference failed: {type(error).__name__}', flush=True)
            self.respond(500, {'error': 'Neural decoding failed. Retry with a shorter recording.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8787)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--engine', choices=['rnnt', 'cwformer'], default='rnnt')
    parser.add_argument('--checkpoint', help='Local checkpoint; uses the validated local release when available.')
    parser.add_argument('--published', action='store_true', help='Use the original published RNN-T model.')
    parser.add_argument('--confidence-threshold', type=float, help='RNN-T acoustic emission threshold, 0–1.')
    parser.add_argument('--digit-threshold', type=float, help='RNN-T digit emission threshold, 0–1.')
    args = parser.parse_args()
    device = args.device
    engine_name = args.engine
    if args.published and (args.checkpoint or args.engine != 'rnnt'):
        parser.error('--published cannot be combined with --checkpoint or --engine cwformer.')
    if args.engine == 'cwformer' and args.device != 'cpu':
        parser.error('The CWformer ONNX engine currently uses CPU inference.')
    if args.engine == 'cwformer' and (args.confidence_threshold is not None or args.digit_threshold is not None):
        parser.error('Acoustic emission thresholds apply to the RNN-T engine only.')
    torch.set_num_threads(4)
    try:
        engines[engine_name] = (load_rnnt(args.checkpoint, args.published, args.confidence_threshold, args.digit_threshold)
                                if engine_name == 'rnnt' else load_cwformer(args.checkpoint))
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
        parser.error(str(error))
    # Optional engines are local additions: a missing artifact must not prevent listening.
    try:
        if engine_name == 'rnnt':
            from cwformer_engine import DEFAULT_MODEL
            if DEFAULT_MODEL.is_file():
                engines['cwformer'] = load_cwformer()
        elif RELEASE_CHECKPOINT.is_file():
            engines['rnnt'] = load_rnnt()
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
        print(f'Optional decoder unavailable: {error}', file=sys.stderr, flush=True)
    loaded = ', '.join(f"{name}: {entry['checkpoint']} ({entry['device']})" for name, entry in engines.items())
    print(f'Neural CW engine ready on http://127.0.0.1:{args.port}; default {engine_name}; {loaded}.', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
