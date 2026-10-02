# cw/boi

A local Morse receiver and audio sender built for irregular hand keying, fading, interference, and continuous radio audio. The browser runs an adaptive DSP decoder; an optional local Python service adds the fine-tuned RNN-T engine or an experimental causal CWformer engine. No dictionary or word language model rewrites the copy.

## Run

```sh
bun install
bun run dev
```

Open the URL Vite prints, normally http://localhost:5173. The adaptive decoder works without Python and runs in an AudioWorklet for live input or a worker for files. Choose your microphone/radio interface under Receiver settings. Use manual carrier tuning to stay on one station in a crowded band.

For acoustic decoding, start a second terminal:

```sh
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r neural/requirements.txt
bun run neural
```

Start the service before opening the page, or refresh after starting it, then select its model in the Decoder menu. In this workspace, the default is **CW boi · RNN-T v1** from `models/cw-boi-rnnt-v1.pt`, checked against [neural/release.json](neural/release.json). Weights are excluded from Git: a fresh checkout without that file automatically downloads the pinned published Morseformer model. Use `bun run neural --published` to select the published weights with the current streaming wrapper, or `--checkpoint path/to/model.pt` for an experiment. RNN-T has approximately four seconds of lookahead. Reproducing the original published decoder requires `neural/benchmark.py --decoder original`; `--published` only changes the weights.

The service binds to `127.0.0.1:8787`; Vite proxies `/api` to it. Audio stays in memory on this computer. CPU inference is supported; the 3080 is useful for training, not required for listening. `--device cuda` needs matching PyTorch and torchaudio CUDA builds. On Windows, invoke `.venv\Scripts\python.exe neural\server.py`.

The optional causal engine runs with `bun run neural --engine cwformer`. It requires the local `models/cwformer-adapt-v4/cwformer_streaming_fp32.onnx` and adjacent mel assets; it does not download them automatically. It processes half-second chunks on CPU and remains experimental. Set **Filter width to 150 Hz** in Receiver settings to match its benchmark; the app starts at 100 Hz for RNN-T. Bandwidth remains adjustable through the UI/API/native CLI. Restart the service to change engines.

## Listen through Terminal on macOS

The native bridge works with a radio interface or BlackHole when browser microphone permission is unavailable. Install FFmpeg, start the neural service, and list the audio inputs:

```sh
brew install ffmpeg
bun run listen --list-inputs
bun run listen --input-index N --frequency 711 --bandwidth 80 --seconds 180
```

Replace `N` with the input's audio index. For BlackHole, route the source application's audio to the same BlackHole device. Allow Terminal microphone access in System Settings → Privacy & Security → Microphone. The bridge prints incremental text, keeps audio in memory, and flushes pending copy on Ctrl-C. Tune the frequency to the actual tone; 711 Hz is an example. Sessions are bounded to 10 minutes.

## Receiver and sender

- Microphone/interface input, local recordings, spectrum/waterfall, automatic or manual carrier tuning, adjustable filtering and signal gate.
- Adaptive mark and gap timing follows speed changes, uneven dah/dit ratios and hesitations. Local spectral noise estimates and carrier prominence reduce copy from colored hiss and static.
- RNN-T reconciles overlapping-window token timestamps and bounds initial digital silence, reducing lost or duplicated characters at window boundaries.
- Audio sender at 5–60 WPM, Farnsworth spacing, joined prosigns, WAV export, loopback, and a keyboard/pointer straight key. Escape stops the station; clearing the transcript stops its session.
- Audio imports: at most 10 minutes and 50 MB. Sending: at most 1,000 characters and 10 minutes.

The sender produces audio; hardware PTT/keying requires a separate interface.

## Measured behavior

On the frozen October 2 continuous synthetic test, RNN-T v1 reduces CER from **18.12% to 15.12%**, and rough-fist CER from **30.77% to 18.30%**, versus the published model with its original streaming decoder. Noise false copy falls from **77 to 19 characters in eight minutes**. The updated adaptive decoder scores **15.56% CER** with zero false characters on those same synthetic negatives.

Two independently transcribed real operators total only **30.4 seconds / 37 characters**: RNN-T makes 6 errors versus the original model's 9. This is a useful check, not evidence of broad real-world superiority. Severe combined distortion still produces **72.06% CER** for RNN-T; neither engine reliably recovers erased information. Confidence and spectral SNR are diagnostics, not calibrated correctness probabilities.

See [full results, reproduction commands and training notes](benchmarks/README.md), and [recording provenance and limitations](benchmarks/CORPUS.md). Final evaluation data must not become checkpoint-selection data.

## Verify

```sh
bun test
bun run build
.venv/bin/python neural/check.py
.venv/bin/python neural/check_streaming.py
.venv/bin/python neural/check_listen.py
.venv/bin/python neural/check_cwformer.py
```

The optional CWformer check requires its model assets. The current suite has 22 passing Bun tests; browser sender → AudioWorklet → RNN-T and adaptive loopback both recovered `CQ TEST 73`. Terminal/FFmpeg captured actual BlackHole audio, but its transcript is unverified. Both HTTP engines and the bridge passed integration checks with simulated capture. A complete live-device bridge session remains unvalidated.

Model, code and recording licenses: [THIRD_PARTY.md](THIRD_PARTY.md).
