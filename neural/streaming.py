"""Reconcile acoustic token timestamps at overlapping decoder windows."""

import numpy as np
import torch
from morseformer.core.tokenizer import decode
from morseformer.decoding.streaming import StreamingDecoder as BaseStreamingDecoder
from morseformer.features import extract_features


class StreamingDecoder(BaseStreamingDecoder):
    def __init__(self, model, cfg, device='cpu'):
        super().__init__(model, cfg, device)
        self._boundary = []
        self._preroll_samples = 0

    def feed(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        if self._total_samples == 0 and audio.size:
            # Keep filter-onset context without letting long digital silence dominate normalization.
            limit = round(self.cfg.sample_rate * 0.2)
            first = int(np.argmax(audio != 0))
            if audio[first] == 0:
                self._preroll_samples = min(limit, self._preroll_samples + audio.size)
                return []
            prefix = min(limit, self._preroll_samples + first)
            audio = np.concatenate((np.zeros(prefix, dtype=np.float32), audio[first:]))
        return super().feed(audio)

    def _commit_tokens(self, events, lo, hi):
        # Independent windows can move an emission across their shared cutoff.
        tolerance = self._enc_samples_per_frame * 4
        previous = self._boundary
        used = set()
        kept = []
        boundary = []
        for token, position in events:
            matches = [(abs(old_position - position), index)
                       for index, (old_token, old_position, _) in enumerate(previous)
                       if index not in used and old_token == token and abs(old_position - position) <= tolerance]
            index = min(matches)[1] if matches else None
            match = previous[index] if index is not None else None
            if index is not None:
                used.add(index)
            committed = bool(match and match[2])
            crossed_left = bool(match and match[1] >= lo and position >= lo - tolerance)
            if not committed and position < hi and (position >= lo or crossed_left):
                kept.append(token)
                committed = True
            if abs(position - hi) <= tolerance:
                boundary.append((token, position, committed))
        self._boundary = boundary
        self._committed_until_samples = hi
        return decode(kept, strip=False)

    def _decode_and_commit(self, *, window_start_samples, window_audio, is_first, is_final):
        features = extract_features(window_audio, self.cfg.sample_rate, self._fcfg)
        if not features.size:
            return ''
        x = torch.from_numpy(features).unsqueeze(0).to(self.device)
        lengths = torch.tensor([features.shape[0]], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            aligned = self.model.greedy_rnnt_decode_aligned(
                x, lengths, max_emit_per_frame=self.cfg.max_emit_per_frame,
                confidence_threshold=self.cfg.confidence_threshold,
                digit_threshold=self.cfg.digit_threshold,
            )[0]
        lo, hi = self._commit_zone_samples(
            window_start_samples=window_start_samples, window_audio_size=window_audio.size,
            is_first=is_first, is_final=is_final,
        )
        events = [(token, window_start_samples + frame * self._enc_samples_per_frame) for token, frame in aligned]
        return self._commit_tokens(events, lo, hi)


def decode_offline(model, audio, cfg, device='cpu'):
    decoder = StreamingDecoder(model, cfg, device)
    return (''.join(decoder.feed(np.asarray(audio, dtype=np.float32))) + decoder.flush()).strip()
