# CWformer on the visitor's device

The hosted app uses the released CWformer v6 fp32 model unchanged, with ONNX Runtime Web 1.30.0 in a single-threaded Web Worker. The model SHA256 is `8344b39dba22e595ea8170bd4ffdb514aaef1c4b67b7de3ef965b5f66a081374`. Its 83,222,226 bytes are divided into four byte-identical assets to fit Cloudflare's per-file limit. Audio filtering, mel features, cache management and transcription all execute in the visitor's browser.

All model inference, automated browser checks and builds below ran on Chonkus. No model inference or build ran on the thermally constrained development Mac.

## Verification

- **36 unit tests, 1,060 assertions passed.** Includes adaptive decoding, frontend DFT/overlap checks, per-head cache truncation, browser client download/retry, stream draining, bounded backlog, cancellation and file abort handling.
- **TypeScript and the production build passed.** All published assets are at most 20 MiB. The main bundle contains neither the legacy `/api/` client nor a localhost inference address. [Build and source hashes](browser-hosting-build.json).
- **30 Python/browser parity checks passed.** Seven previously consumed synthetic fixtures cover clean CW, irregular human-style timing, rough and chaotic conditions, digital silence, loud Gaussian noise and an 80-second interrupted transmission. These fixtures total 211.0075 seconds of audio. Every transcript and emission boundary matched Python, including existing decoding mistakes. Incoming blocks of 137, 4,000, 8,011 and whole-file lengths, a retune with pending audio, repeated flush and nonfinite input were checked. [Parity report](browser-cwformer-parity.json), [reference output and waveform hashes](browser-cwformer-reference.json).
- **Numerical frontend parity passed at 80, 100 and 150 Hz bandwidths**, including a 650→657 Hz tune adjustment. Maximum sample error was `7.28e-12`; maximum mel-feature error was `9.54e-7`.
- **The actual production worker bundle passed live and file checks**, including model-part integrity checks, initialization, download/decode progress, repeated finish and exact Python emission boundaries. Only same-origin GET requests occurred. [Production worker report](browser-cwformer-production.json).
- **The deployed site passed three worker checks:** live copy, file copy and cancellation after inference had started, with no late result. All 19 observed requests were same-origin GETs; there were no audio uploads, API calls or analytics requests. [Hosted report](browser-cwformer-hosted.json).

After the recorded build, only the response-header asset was amended to prevent Cloudflare from injecting its automatic analytics beacon into the app HTML. The final `public/_headers` SHA256 is `c04e92ccbd94e12be2b9169968c53ac5e50c91f9bb983542b2671bee9cb5eee6`; JavaScript, worker and model assets were unchanged. The hosted check above ran against deployment `a47561e0-4382-4b16-a3b8-e95e091a21ed` after that change.

The production worker processed 19.3635 seconds of audio in 2.605 seconds on the tested Windows Chrome 154 machine, including stream warm-up. This is a real-time factor of 0.135 on that device. It is not a speed guarantee for phones or other computers. The measured 1.195-second model load used a local HTTP server and is not an internet download estimate.

These are implementation checks, not a new accuracy benchmark. “Human-style” here means synthetic irregular timing; no new human recording or reserved final corpus was used. Severe-noise and pause errors from the Python model remain. The noise-only check covers 20 seconds of loud white noise plus 20 seconds of digital silence, not all receiver interference.

## Reproduction

Run on an evaluation machine with the released model, the existing consumed `round3-dev` and `round3-interrupted-dev` waveform directories, Python neural dependencies, Bun, Node and installed Chrome. The reference exporter validates every selected waveform's recorded checksum.

```sh
python neural/export_browser_reference.py --data-root <consumed-data-root> --model models/cwformer-weighted-v6/step-2000/cwformer_streaming_fp32.onnx --out .research/browser-reference
python scripts/prepare-browser-model.py
bun install --frozen-lockfile --ignore-scripts
bun test
bun run build
```

For core parity, start Vite with Node (Cloudflare's TypeScript configuration loader requires Node) and run the browser runner in another terminal. Set `CW_CHROME` to the installed Chrome executable first.

```sh
node node_modules/vite/bin/vite.js --host 127.0.0.1 --port 5179
node scripts/check-browser-cwformer.mjs http://127.0.0.1:5179/scripts/check-browser-cwformer.html benchmarks/browser-cwformer-parity.json
```

For the exact built worker, start `node scripts/check-browser-cwformer-server.mjs`, then point the runner at `http://127.0.0.1:5180/__checks__/check-browser-cwformer-production.html?worker=/assets/worker-KDxYxdwP.js`. Substitute the worker filename from the current build. The helper serves test fixtures only on loopback; it is not part of the deployment.

For a deployed-site check, set `CW_PRODUCTION_WORKER` to that same worker asset path and run the browser runner against the site's root URL. It injects the consumed synthetic fixture through the local browser debugging pipe, tests stream/file inference and cancellation after decoding starts, and rejects any external requests, POST requests or `/api/` traffic. No diagnostic endpoint or test audio is published.
