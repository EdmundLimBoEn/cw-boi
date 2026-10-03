# cw/boi

A local Morse receiver and audio sender built for irregular hand keying, fading, interference, and continuous radio audio. The browser runs an adaptive DSP decoder; an optional local Python service serves the fine-tuned RNN-T and causal CWformer engines. No dictionary or word language model rewrites the copy.

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

Start the service before opening the page, or refresh after starting it, then choose **Adaptive signal decoder**, **CW boi · RNN-T v1**, or **CWformer · v6** in the Decoder menu or Signal lab. **CWformer v6 is the recommended installed model** after the October 3 evaluation. One service serves both installed neural engines; switching in the browser stops the current session and clears its copy and lab results. The CLI's compatibility default remains **CW boi · RNN-T v1** from `models/cw-boi-rnnt-v1.pt`, checked against [neural/release.json](neural/release.json). Weights are excluded from Git: a fresh checkout without that file automatically downloads the pinned published Morseformer model. Use `bun run neural --published` to select the published weights with the current streaming wrapper, or `--checkpoint path/to/model.pt` for an experiment. RNN-T has approximately four seconds of lookahead. Reproducing the original published decoder requires `neural/benchmark.py --decoder original`; `--published` only changes the weights.

The service binds to `127.0.0.1:8787`; Vite proxies `/api` to it. The service keeps audio in memory without saving or logging it. CPU inference is supported; the 3080 is useful for training, not required for listening. `--device cuda` needs matching PyTorch and torchaudio CUDA builds. On Windows, invoke `.venv\Scripts\python.exe neural\server.py`.

CWformer is offered automatically when `models/cwformer-weighted-v6/step-2000/cwformer_streaming_fp32.onnx` and its adjacent mel assets are available; the service does not download them automatically. Missing or unloadable optional weights leave the default engine available. CWformer processes half-second chunks on CPU. Selecting it sets **Filter width to 150 Hz**; selecting RNN-T sets **100 Hz**. Bandwidth remains adjustable through the UI/API/native CLI.

`--engine rnnt|cwformer` chooses the default for clients that omit an engine, including the Terminal bridge; browser requests select their engine explicitly. `--checkpoint` applies to that default engine. With the local v6 weights installed, use **`bun run neural --engine cwformer`** to make the bridge use CWformer while the browser can still choose either installed model. No service restart is needed to switch models in the browser.

### Keep neural inference off the Mac

The October 3 working session runs the neural service on Chonkus with two CPU threads; the interface, audio capture and spectrum stay on the Mac. Audio crosses the encrypted SSH connection to that user-owned machine and stays in its service memory. Chonkus must remain awake while using either neural engine. The adaptive browser decoder still works without it.

With the remote service running, forward its loopback port into the Mac's existing Vite proxy:

```sh
ssh -NT -L 127.0.0.1:8787:127.0.0.1:8787 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o HostKeyAlias=chonkus limbo@192.168.0.17
```

The address is Chonkus's current LAN address. Keep the tunnel terminal open; reconnect it after a disconnect, then refresh the app. Do not start a local neural service on the same port while the tunnel is active. Exact remote launch settings and artifact hashes are recorded in [the service record](benchmarks/round3-service.json).

## Listen through Terminal on macOS

The native bridge works with a radio interface or BlackHole when browser microphone permission is unavailable. Install FFmpeg, start the neural service, and list the audio inputs:

```sh
brew install ffmpeg
bun run listen --list-inputs
bun run listen --input-index N --frequency 711 --seconds 180
```

Replace `N` with the input's audio index. For BlackHole, route the source application's audio to the same BlackHole device. Allow Terminal microphone access in System Settings → Privacy & Security → Microphone. The bridge prints incremental text, keeps audio in memory, and flushes pending copy on Ctrl-C. Tune the frequency to the actual tone; 711 Hz is an example. Sessions are bounded to 10 minutes. Filter width follows the selected engine (CWformer 150 Hz, RNN-T 100 Hz); use `--bandwidth` to override it. Narrower filters are not automatically better: the CWformer weak/noisy development probes performed worse at 80–120 Hz than at 150 Hz.

## Receiver and sender

- Microphone/interface input, local recordings, spectrum/waterfall, automatic or manual carrier tuning, adjustable filtering and signal gate.
- Adaptive mark and gap timing follows speed changes, uneven dah/dit ratios and hesitations. Local spectral noise estimates and carrier prominence reduce copy from colored hiss and static.
- RNN-T reconciles overlapping-window token timestamps and bounds initial digital silence, reducing lost or duplicated characters at window boundaries.
- Audio sender at 5–60 WPM, Farnsworth spacing, joined prosigns, WAV export, loopback, and a keyboard/pointer straight key. Escape stops the station; clearing the transcript stops its session.
- Audio imports: at most 10 minutes and 50 MB. Sending: at most 1,000 characters and 10 minutes.

The sender produces audio; hardware PTT/keying requires a separate interface.

## Measured behavior

The latest automatic-tuning fix passes all 16 frozen integration gates. In a separate synthetic acquisition test, live errors fall from **430 to 2 in 760 characters**, and imported-file errors from **452 to 0 in 696**. All quiet carriers and established crowded-channel targets are retained; false copy stays at **six characters per eight minutes** of Gaussian noise for each path. These fixtures test quiet high-SNR acquisition and station tracking, not unseen human operators or extreme RF noise. [Protocol and limits](benchmarks/ROUND3_RESULTS.md).

On the frozen October 3 test, **CWformer v6 scores 9.88% character error rate**, versus **14.83% for RNN-T v1** and **10.18% for CWformer v4**, across 3,027 synthetic reference characters. Noise false copy is **0 versus RNN-T's 17 characters in eight minutes**. On a separately reserved real reception, v6 makes **7 errors in 422 characters**, versus 11 for both earlier neural engines. It passed every predeclared promotion guard.

Short-gap training targets heavily weighted hand keying. Development recordings improve from **99 to 16 errors in 438 characters** versus CWformer v4; those recordings were used for selection, not final proof. The old 37-character Marine regression still favors RNN-T: 6 errors versus v6's 12. The new real final is one reception session, not nine independent operators. Severe combined distortion still produces **51.15% CER** for v6; confidence and spectral SNR are diagnostics, not calibrated correctness probabilities.

The earlier proposed adaptive timing change improved real recordings but tied its synthetic final baseline, failing the strict improvement guard; that patch remains archived. The latest scanner fix applies to carrier acquisition shared by adaptive and neural decoding. Causal-front-end retraining, additional noise training, CTC beam decoding and proposed pause resets did not qualify for deployment.

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

The optional CWformer check requires its model assets. The current suite has 25 passing Bun tests and a verified production build, run on Chonkus after the Mac heat report. Both neural engines recover the transport smoke-test recording exactly through the Mac Vite proxy and SSH forward. Earlier browser sender → AudioWorklet → RNN-T, CWformer and adaptive loopback recovered `CQ TEST 73`. Terminal/FFmpeg captured actual BlackHole audio, but its transcript is unverified. A complete live-device bridge session remains unvalidated.

Model, code and recording licenses: [THIRD_PARTY.md](THIRD_PARTY.md).
