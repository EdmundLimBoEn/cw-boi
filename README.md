# cw/boi

A Morse receiver and audio sender built for irregular hand keying, fading, interference, and continuous radio audio. The hosted app runs both the adaptive DSP decoder and CWformer v6 entirely on the user's device. Audio stays in the browser; no Python service, SSH tunnel, or Chonkus connection is needed. No dictionary or word language model rewrites the copy.

## Use the app

Open **[cw-boi.sillyapps.co](https://cw-boi.sillyapps.co)** in a browser with WebAssembly and AudioWorklet support. Allow microphone access when you start listening, then choose your microphone or radio interface under Receiver settings. For BlackHole, route the radio application's output to the same BlackHole input.

**CWformer v6 · on device** downloads its model only when you load it or start a neural decoding action. The first load transfers approximately **97 MB** of model and WebAssembly files; progress, cancellation and retry are available. Inference runs in a dedicated browser worker using one WebAssembly thread. The browser can cache the files for later visits, but this is not a fully offline application.

Choose **Adaptive signal decoder** for the lighter option: it requires no model download and releases the neural worker. Both decoders, the sender and local recording imports work without sending audio to a server. Processing still uses your device's CPU; Stop halts an active session. Use manual carrier tuning to stay on one station in a crowded band. CWformer starts with a 150 Hz filter; narrower filters are not automatically better.

## Develop locally

```sh
bun install
bun run dev
```

Open the URL Vite prints, normally http://localhost:5173. The adaptive decoder works immediately. CWformer also needs the prepared public model assets, which are excluded from Git. With the v6 release model available locally:

```sh
python3 scripts/prepare-browser-model.py --source models/cwformer-weighted-v6/step-2000 --destination public/models/cwformer-v6
bun run assets:prepare
```

This copies the exact FP32 weights into hashed chunks below Cloudflare's 25 MiB asset limit and converts the mel assets. It does not train, quantize or run the model. `assets:prepare` verifies the model and mel hashes and copies the pinned ONNX Runtime Web files from `node_modules/onnxruntime-web/dist`. Keep those generated files available for development and deployment.

## Deploy to Cloudflare

The project uses Workers Static Assets and the project-local `cf` CLI, configured in [cloudflare.config.ts](cloudflare.config.ts). The custom domain is `cw-boi.sillyapps.co`; Cloudflare manages its DNS record and HTTPS certificate. There is no inference API or audio storage at the host.

Use Node.js 22.18 or newer to run `cf`; Bun remains the package manager. The project pins its CLI and Cloudflare Vite plugin because their configuration format is in beta. The machine's older global `cf` installation may use a separate login.

```sh
node node_modules/cf/bin/cf auth login
bun run build
node node_modules/cf/bin/cf deploy --prebuilt --mode production --dry-run
node node_modules/cf/bin/cf deploy --prebuilt --mode production
```

Prepare the model and runtime assets before building. Deployment reuses the verified production output, including its `production` mode. The account and custom domain are explicit in the project config, so deployment does not depend on a global zone selection. No Wrangler configuration is used.

## Optional Python service and research tools

The Python service is retained for Terminal capture, RNN-T experiments and reproducing earlier benchmarks. It is independent of the hosted browser app; starting it does not add RNN-T to the hosted decoder menu.

Set up and run the service:

```sh
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r neural/requirements.txt
bun run neural
```

The CLI's compatibility default is **CW boi · RNN-T v1** from `models/cw-boi-rnnt-v1.pt`, checked against [neural/release.json](neural/release.json). Weights are excluded from Git: a fresh checkout without that file automatically downloads the pinned published Morseformer model. Use `bun run neural --published` to select the published weights with the current streaming wrapper, or `--checkpoint path/to/model.pt` for an experiment. RNN-T has approximately four seconds of lookahead. Reproducing the original published decoder requires `neural/benchmark.py --decoder original`; `--published` only changes the weights.

The service binds to `127.0.0.1:8787` and keeps audio in memory without saving or logging it. The hosted app and current Vite app do not call it. CPU inference is supported; `--device cuda` needs matching PyTorch and torchaudio CUDA builds. On Windows, invoke `.venv\Scripts\python.exe neural\server.py`.

CWformer is available to the service when `models/cwformer-weighted-v6/step-2000/cwformer_streaming_fp32.onnx` and its adjacent mel assets are installed; the service does not download them automatically. Use **`bun run neural --engine cwformer`** to select it for the Terminal bridge. The service uses 150 Hz for CWformer and 100 Hz for RNN-T unless the client overrides the bandwidth.

### Keep neural inference off the Mac

For Python/Terminal work, the October 3 setup runs the service on Chonkus with two CPU threads. Forward its loopback port to the Mac only when using those tools:

```sh
ssh -NT -L 127.0.0.1:8787:127.0.0.1:8787 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o HostKeyAlias=chonkus limbo@192.168.0.17
```

Audio sent through this bridge crosses the encrypted SSH connection to that user-owned machine and stays in its service memory. Keep Chonkus awake and the tunnel terminal open. Do not start a local Python service on the same port while the tunnel is active. The current LAN address, launch settings and artifact hashes are recorded in [the service record](benchmarks/round3-service.json). This setup is optional and unrelated to the on-device web deployment.

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
- CWformer v6 processes half-second audio chunks locally; the separate Python RNN-T tool retains its overlapping-window timestamp reconciliation.
- Audio sender at 5–60 WPM, Farnsworth spacing, joined prosigns, WAV export, loopback, and a keyboard/pointer straight key. Escape stops the station; clearing the transcript stops its session.
- Audio imports: at most 10 minutes and 50 MB. Sending: at most 1,000 characters and 10 minutes.

The sender produces audio; hardware PTT/keying requires a separate interface.

## Measured behavior

The following accuracy results describe the original Python/ONNX evaluation pipeline. The browser port ships the same FP32 model; these historical results are not a separate browser or hardware performance benchmark.

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

The browser port passes 36 Bun tests (1,060 assertions), a production build, and 30 Python/browser parity checks on Chonkus. Its actual built worker also passes streaming and file decoding checks using the exact released model. [Browser verification and reproduction](benchmarks/BROWSER_PORT.md).

The optional Python CWformer check requires its model assets. Before the browser port, both Python engines recovered the transport smoke-test recording exactly through the former Vite API proxy and SSH forward. Earlier browser sender → AudioWorklet → RNN-T, CWformer and adaptive loopback recovered `CQ TEST 73`. Terminal/FFmpeg captured actual BlackHole audio, but its transcript is unverified. A complete live-device bridge session remains unvalidated.

Model, code and recording licenses: [THIRD_PARTY.md](THIRD_PARTY.md).
