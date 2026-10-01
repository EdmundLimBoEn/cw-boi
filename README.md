# cw/boi

A local CW receiver and audio sender, with two independent decoding paths and a reproducible signal laboratory. React + TypeScript; DSP runs in an AudioWorklet for live audio and a worker for recordings. The optional Python acoustic engine uses the published Morseformer model.

## Run

```sh
bun install
bun run dev
```

Open the URL Vite prints, normally http://localhost:5173. The adaptive decoder works without Python or an internet connection after installation. Microphone access requires localhost or HTTPS; choose your radio audio interface under Receiver settings. Disable automatic tuning to stay with one station in a crowded band.

For neural decoding, in a second terminal:

```sh
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r neural/requirements.txt
bun run neural
```

On Windows use `.venv\Scripts\python.exe neural\server.py`. Start the service before opening the page, or refresh the page after it starts, then choose Neural · Morseformer in the Decoder menu. The first run downloads the pinned model; subsequent inference stays local. The service binds only to `127.0.0.1:8787`, accepts local origins, and keeps audio in memory. Vite proxies `/api` to it. `--device cuda` and `--checkpoint path/to/model.pt` are optional server arguments. CUDA needs a matching PyTorch/torchaudio installation; CPU inference is supported and does not need the 3080.

## What works

- Microphone/audio interface input, local audio files, automatic or manual carrier tuning, a live spectrum/waterfall, adjustable filtering and signal gate.
- Adaptive dot/dash and spacing estimation with local timing clusters and several competing Morse interpretations; carrier confirmation rejects noise-only excursions.
- Optional acoustic Conformer/RNN-T decoding with about four seconds of lookahead. No optional word language model or dictionary rewrites are enabled. Learned acoustic models can still hallucinate.
- Audio sending at 5–60 WPM, Farnsworth spacing, adjustable tone and volume, joined prosigns, WAV export, internal loopback, and a keyboard/pointer straight key. Escape stops the station. Clearing the transcript also stops its current session.
- Recordings up to 10 minutes and 50 MB. Neural live sessions have a 10-minute limit. Sending supports 1,000 characters, bounded by 10 minutes.

The sender produces **audio**, not a hardware PTT/keying signal. Connect a suitable interface separately if you need transmitter control.

## Verification and limits

```sh
bun test
bun run build
bun run benchmark
bun run scripts/compare-decoders.ts
.venv/bin/python neural/check.py
```

The comparison command requires the neural service. Full measured outputs and GPU results are in [benchmarks/README.md](benchmarks/README.md). The fine-tuned model remains experimental because its gains did not fully transfer to continuous decoding. Character error rate includes substitutions, insertions, deletions, and spaces; it can exceed 100% when a decoder invents extra characters. SNR in this generator is keyed carrier RMS versus broadband noise RMS at 8 kHz, not a claim about receiver bandwidth or an RF measurement.

The saved adaptive run gets **32/48 exact messages and 9.64% aggregate character error**. The regression suite has 48 seeded random-message cases covering 10–40 WPM, up to 35% timing jitter, 2.6–4.4 dah/dit ratios, speed swing, drift, fading, and SNR down to −5 dB. The six comparison cases also include key bounce, hesitation, abrupt speed changes, another actual Morse station, flutter, and static crashes. Unit checks separately cover clean 8–50 WPM copy, Farnsworth timing, noise rejection, and streaming block sizes.

**These are synthetic tests, not evidence of best-in-class real-world performance.** Combined severe noise and irregular timing remains a significant failure case. Neither decoder can reconstruct information that has been erased; use the displayed hypotheses as copy to verify, especially callsigns and numbers. Confidence values and spectral SNR are diagnostic estimates, not calibrated probabilities of correctness.

To make a defensible quality claim, the next evaluation corpus needs licensed, independently transcribed real recordings split by operator, radio, and session, including noise-only clips. Compare with established decoders on the same recordings, measure character error, false text per minute, acquisition time, and latency, and publish failures alongside successes. Do not train on that final test corpus.

## Bounded fine-tuning

`neural/train.py` fine-tunes the pinned published model; it does not train a new model from scratch. Its independent Python generator emits complete, untruncated 6-second examples with random messages, rough timing, speed changes, fading, drift, noise, static, adjacent CW, and noise-only targets. A single CPU smoke step is:

```sh
.venv/bin/python neural/train.py --device cpu --steps 1 --batch-size 2 --workers 0 --eval-every 1 --eval-count 4 --output models/smoke
```

For the 10 GB RTX 3080, install matching CUDA builds of `torch==2.8.0` and `torchaudio==2.8.0`, then `morseformer==0.6.4`, in an isolated Python 3.12 environment. Run:

```sh
python neural/train.py --device cuda --steps 600 --batch-size 4 --workers 2 --eval-every 200 --eval-count 48 --output models/rough-fist-v1
```

Training and validation use separate fixed seeds. `baseline.json`, `eval-N.json`, and `last.pt` are always written. `best.pt` is written only when chaotic-signal validation error improves, clean/rough error worsens by no more than two percentage points, and noise-only false characters do not increase. This is a development selection gate, not a final accuracy guarantee. Check the selected checkpoint on a further untouched seed and the independent TypeScript generator before deployment. The UI does not automatically adopt a training checkpoint. An independent final evaluation uses a seed never used for checkpoint selection:

```sh
python neural/train.py --device cuda --evaluate-only --eval-seed 3000000 --eval-count 128 --checkpoint models/rough-fist-v1/best.pt --output models/final-holdout
```

Compare that report against the same command without `--checkpoint` (the published model). `--hard-fraction 0.45` increases combined-distortion training examples while retaining noise-only, clean, and rough examples.

Model/code attribution and licensing: [THIRD_PARTY.md](THIRD_PARTY.md).

To explicitly try the experimental local checkpoint instead of the published model, stop the neural service and run:

```sh
bun run neural --checkpoint models/rough-fist-gpu-v2/best.pt
```

The checkpoint is available in this workspace and on chonkus, not in Git. Restart `bun run neural` without that argument to return to the published model.
