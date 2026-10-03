# Round 3 acquisition and noise investigation

The shared carrier acquisition fix passes all 16 frozen integration gates and is deployed. CWformer weighted v6 remains the acoustic model: both bounded training runs completed, but none of their six checkpoints passed the development gates. Neural inference now runs on Chonkus through SSH to keep that load off the Mac.

## Carrier acquisition

The scanner previously rejected quiet, high-SNR tones below an absolute power threshold. It could also jump to a louder adjacent station during the desired station's key-up gaps. The candidate removes the redundant absolute-power checks while retaining spectral prominence checks, and reuses the existing 1.2-second confirmed-carrier hold before allowing a distant retune. Manual tuning remains available.

The [frozen pipeline](round3-acquisition-freeze.json) uses the original v6 model and neural runtime on both sides. Final seed `4401901` was generated only after candidate freeze and approval to evaluate. The 60 fixtures contain 36 speech recordings and eight minutes of Gaussian noise, 991.77 seconds total. Four changing-carrier recordings are live-only, because file decoding selects one carrier for the whole recording.

| Fresh final path | Reference characters | Baseline errors | Fixed scanner errors | Noise false characters / 8 minutes, baseline → fixed |
| --- | ---: | ---: | ---: | ---: |
| Live queue replay | 760 | 430 (56.58%) | 2 (0.263%) | 6 → 6 |
| Imported-file replay | 696 | 452 (64.94%) | 0 | 6 → 6 |

Both paths acquire all 16 quiet carriers and retain all eight established crowded-channel targets. All four new stations are acquired within 0.294–0.362 seconds of the new recording's start. Normal clean copy remains exact. The [final comparison](round3-acquisition-final-comparison.json) applies every gate unchanged; its linked reports retain every prediction. [Commands](round3-acquisition-final-commands.json) and [descriptive paired uncertainty](round3-acquisition-final-uncertainty.json) preserve reproduction details. No tuning or candidate reselection followed the final result.

The earlier [development replay](round3-acquisition-dev-comparison.json) scored 336 → 3 live errors and 367 → 0 file errors on the same reference counts, with unchanged five/four false characters per four minutes. The final set is now consumed and must not be reused to select another scanner change.

The replay exercises the actual `NeuralStream` queue, sample-before-reading worklet ordering, and separate 256-sample file scanning. It assumes zero transport latency and excludes browser resampling and physical audio capture. Fixed message templates are reused with fresh waveform parameters. Quiet high-SNR acquisition is distinct from decoding CW buried in RF noise; this suite does not establish new human-operator or severe-noise accuracy.

## Matched training result

Two RTX 3080 runs start from the same original v6 checkpoint, use the same seed, batch size four, learning rate 1e-5, and 3,000 steps. Each has a 900-second limit. The treatment adds 3 dB of Gaussian noise only to chaos-family segments that do not use recorded interference. Short-gap augmentation, other impairments, labels and recorded backgrounds stay unchanged. The default training option is verified bit-identical on 32 waveforms and eight complete training items.

| Checkpoint | Standard errors / 1,459 | Rough + chaos errors / 455 | False characters / 4 minutes |
| --- | ---: | ---: | ---: |
| Original v6 | 134 | 132 | 1 |
| Control, 1,000 | 136 | 134 | 0 |
| Control, 2,000 | 136 | 132 | 2 |
| Control, 3,000 | 129 | 127 | 2 |
| Harder noise, 1,000 | 133 | 132 | 0 |
| Harder noise, 2,000 | 129 | 127 | 4 |
| Harder noise, 3,000 | 128 | 126 | 2 |

The best treatment reduces primary errors by 4.55%, below the predeclared 10% minimum, and increases noise false copy. Its advantage over the matched control is one error at step 3,000. No checkpoint is promoted. Most interrupted/real checks are deliberately skipped after a decisive standard-development failure; the completed control-1,000 checks remain in the record. This is one paired training seed, not proof that other noise-training approaches cannot help.

[Run protocol and compatibility](training-cwformer-noise-round3/protocol.json), [all checkpoint decisions and per-clip reports](training-cwformer-noise-round3/comparison.json), and [diagnostic ablations](round3-model-diagnosis.json) retain the evidence. All six ONNX exports were checked against Torch. The remote two-thread evaluator reproduces every original standard-development prediction; interrupted and consumed-real baseline aggregates also match. Production inference thread settings are unchanged.

## Other measured decisions

- **Keep greedy CTC decoding.** Prefix beam four changes consumed synthetic errors from 148 to 147, with worse rough-fist copy. Real development errors improve from 16 to 14 entirely through spacing; non-space errors remain six. It also needs live text-commit handling. [Rejected probe](round3-ctc-probe.json).
- **Keep the current pause runtime.** Removing repeated token-quiet cache resets reduces interrupted errors from 246 to 235 in 596 characters, only 4.47%. Gap false characters improve from 18 to two, but crowded-pause errors rise from 112 to 114. Other regression checks were stopped after the Mac heat report and are not claimed to pass. Fixed gain was diagnostic only because it can hide later weaker stations. [Rejected probe](round3-pause-probe.json).
- **Use the engine's filter recommendation in the Terminal bridge.** It previously forced 80 Hz even for CWformer. Omitting `--bandwidth` now selects CWformer's 150 Hz or RNN-T's 100 Hz; explicit overrides remain supported. Consumed development probes at 80–120 Hz worsened CWformer's combined-chaos errors relative to 150 Hz. The browser advice now reflects that measurement.

The new standard/interrupted neural final seeds and the [sealed NW6V human session](../neural/corpus-final-v3-sealed.json) remain unused, because neither neural lane admitted a candidate. The acquisition lane has its own independent final waveform seed and gates in the [prospective plan](round3-plan.json).

## Verification and thermal handling

Chonkus passes 25 Bun tests, 885 assertions, the production TypeScript/Vite build, and the native bridge checks. Training compatibility and CWformer checks also pass remotely. After the user reported excessive Mac heat, local inference workers were stopped; remaining training, exports, full benchmarks and build verification moved to Chonkus. [Verification record](round3-verification.json).

The Mac runs Vite and a loopback-only SSH forward. The [Chonkus service](round3-service.json) serves the exact v6 and RNN-T v1 releases with two inference threads, with audio kept in memory. Both engines recover an existing 16-second recording exactly through the real Vite → SSH → HTTP path ([smoke check](round3-service-smoke.json)); no audio was synthesized or decoded by a model on the Mac for that check. The browser is idle with both engines available. Actual BlackHole reception still lacks verified reference copy, and HTTP queue behavior under sustained network latency is not part of the acquisition benchmark.
