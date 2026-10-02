"""Causal CWformer ONNX inference with a selected-carrier audio front end.

Mel and cache protocol adapted from parsimo2010/CWformer, MIT; see CWFORMER_LICENSE.
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import butter, firwin, lfilter, sosfilt

TOKENS = [''] + [' '] + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?/(&=+') + ['AR', 'SK', 'BT', 'KN', 'AS', 'CT']
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / 'models/cwformer-adapt-v4/cwformer_streaming_fp32.onnx'


class Model:
    def __init__(self, path=None):
        import onnxruntime as ort
        self.path = Path(path or DEFAULT_MODEL)
        self.config = json.loads((self.path.parent / 'mel_config.json').read_text())
        self.window = np.load(self.path.parent / 'mel_window.npy').astype(np.float32)
        self.basis = np.load(self.path.parent / 'mel_basis.npy').astype(np.float32)
        if self.config['sample_rate'] != 16000 or self.config['hop_length'] != 160 or self.config['n_fft'] != 400:
            raise ValueError('Expected the pinned 16 kHz CWformer front end')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.path), options, providers=['CPUExecutionProvider'])

    def state(self):
        c = self.config
        state = {'pos_offset': np.zeros(1, np.int64),
                 'sub_buf1': np.zeros((1, 1, 2, c['n_mels']), np.float32),
                 'sub_buf2': np.zeros((1, c['subsample_channels'], 2, math.ceil(c['n_mels'] / 2)), np.float32)}
        for layer in range(c['n_layers']):
            for part in ('k', 'v'):
                state[f'kv_{part}_layer{layer}'] = np.zeros((1, c['n_heads'], 0, c['d_model'] // c['n_heads']), np.float32)
            state[f'conv_buf_layer{layer}'] = np.zeros((1, c['d_model'], c['conv_kernel'] - 1), np.float32)
        return state


class Frontend:
    def __init__(self, frequency, bandwidth):
        self.frequency, self.bandwidth = frequency, bandwidth
        self.lowpass = butter(3, bandwidth / 2, fs=8000, output='sos')
        self.filter_state = np.zeros((len(self.lowpass), 2), np.complex128)
        self.resampler = firwin(41, .5, window=('kaiser', 5)) * 2
        self.resample_state = np.zeros(40)
        self.phase = 0.0
        self.output_phase = 0.0
        self.peak = 0.0

    def process(self, audio):
        index = np.arange(len(audio))
        phases = self.phase + index * (2 * np.pi * self.frequency / 8000)
        base, self.filter_state = sosfilt(self.lowpass, audio * np.exp(-1j * phases) * 2, zi=self.filter_state)
        self.phase = (self.phase + len(audio) * 2 * np.pi * self.frequency / 8000) % (2 * np.pi)
        phases = self.output_phase + index * (2 * np.pi * 650 / 8000)
        shifted = (base * np.exp(1j * phases)).real
        self.output_phase = (self.output_phase + len(audio) * 2 * np.pi * 650 / 8000) % (2 * np.pi)
        expanded = np.zeros(len(audio) * 2)
        expanded[::2] = shifted
        result, self.resample_state = lfilter(self.resampler, [1], expanded, zi=self.resample_state)
        self.peak = max(float(np.max(np.abs(result), initial=0)), self.peak * math.exp(-len(audio) / 16000))
        return (result * (.7 / max(self.peak, 1e-5))).astype(np.float32)


class StreamingDecoder:
    def __init__(self, model, cfg, device='cpu'):
        self.model, self.cfg = model, cfg
        self.pending = np.empty(0, np.float32)
        self.previous_token = 0
        self.emitted = False
        self.trailing_space = False
        self.finished = False
        self._reset_signal()

    def _reset_signal(self):
        self.frontend = Frontend(self.cfg.carrier_hz, self.cfg.bandwidth_hz)
        self.state = self.model.state()
        self.overlap = np.zeros(200, np.float32)
        self.quiet_frames = 0
        self.previous_token = 0
        self._infer(np.zeros(16000, np.float32))

    def _retune(self):
        # Pending samples belong to the old passband; do not decode them after a retune.
        retuned = abs(self.frontend.frequency - self.cfg.carrier_hz) > max(12, self.cfg.bandwidth_hz / 3)
        if retuned or self.frontend.bandwidth != self.cfg.bandwidth_hz:
            self.pending = np.empty(0, np.float32)
            self.trailing_space = self.emitted
            self._reset_signal()

    def _chunk(self, audio):
        self.frontend.frequency = self.cfg.carrier_hz
        return self._infer(self.frontend.process(audio))

    def _infer(self, audio):
        joined = np.concatenate((self.overlap, audio))
        count = max(0, (len(joined) - 400) // 160 + 1)
        self.overlap = joined[count * 160:].copy()
        if not count:
            return ''
        frames = np.lib.stride_tricks.sliding_window_view(joined, 400)[::160][:count]
        power = np.abs(np.fft.rfft(frames * self.model.window, axis=1)) ** 2
        mel = np.log(power @ self.model.basis.T + 1e-6).astype(np.float32)[None]
        outputs = self.model.session.run(None, {'mel_chunk': mel, **self.state})
        self.state['pos_offset'] = outputs[1]
        at = 2
        for layer in range(self.model.config['n_layers']):
            for part in ('k', 'v'):
                self.state[f'kv_{part}_layer{layer}'] = outputs[at][:, :, -250:, :]
                at += 1
        for layer in range(self.model.config['n_layers']):
            self.state[f'conv_buf_layer{layer}'] = outputs[at]
            at += 1
        self.state['sub_buf1'], self.state['sub_buf2'] = outputs[at:at + 2]
        text = []
        for token in np.argmax(outputs[0][:, 0], axis=1):
            token = int(token)
            if token > 1:
                self.quiet_frames = 0
            else:
                self.quiet_frames += 1
            if token and token != self.previous_token:
                if token == 1:
                    self.trailing_space = self.emitted
                else:
                    text.append((' ' if self.trailing_space else '') + TOKENS[token])
                    self.emitted = True
                    self.trailing_space = False
            self.previous_token = token
        if self.emitted and self.quiet_frames >= 250:
            self.state = self.model.state()
            self.previous_token = 0
            self.quiet_frames = 0
        return ''.join(text)

    def feed(self, audio):
        if self.finished:
            raise ValueError('Cannot feed a finished decoder')
        audio = np.asarray(audio, np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError('Audio must contain finite mono samples')
        self._retune()
        chunks = []
        needed = 4000 - len(self.pending)
        if len(audio) < needed:
            self.pending = np.concatenate((self.pending, audio))
            return chunks
        if len(self.pending):
            text = self._chunk(np.concatenate((self.pending, audio[:needed])))
            if text:
                chunks.append(text)
            audio = audio[needed:]
        for start in range(0, len(audio) - 3999, 4000):
            text = self._chunk(audio[start:start + 4000])
            if text:
                chunks.append(text)
        self.pending = audio[len(audio) // 4000 * 4000:].copy()
        return chunks

    def flush(self):
        if self.finished:
            return ''
        self._retune()
        text = []
        if len(self.pending):
            text.append(self._chunk(np.pad(self.pending, (0, 4000 - len(self.pending)))))
        text.append(self._chunk(np.zeros(4000, np.float32)))
        self.pending = np.empty(0, np.float32)
        self.finished = True
        return ''.join(text)


def decode_offline(model, audio, cfg, device='cpu'):
    decoder = StreamingDecoder(model, cfg, device)
    return (''.join(decoder.feed(audio)) + decoder.flush()).strip()
