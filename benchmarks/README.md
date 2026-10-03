# Measured results

**Current: [October 3 results and reproduction](ROUND2_RESULTS.md).** CWformer v6 passed all seven release guards against both CWformer v4 and RNN-T v1 and is now the recommended installed neural model. The [deployment record](round2-deployment.json) identifies the exact artifact and post-evaluation path-only wiring change. The adaptive proposal was rejected and archived; its deployed source remains `87ced16`.

## Historical: October 2 release

The remaining results describe the October 2 release and its now-consumed evaluation data. References to "current" in this historical section mean that release.

**RNN-T v1 is the local neural default.** Its frozen checkpoint and runtime passed all six [predeclared release guards](release-criteria.json): lower aggregate synthetic CER, bounded clean/fading/crowded regressions, no increase in noise false copy, and no more than one additional real-recording error. [Selection](rnnt-selection.json), [final comparison](rnnt-final-comparison.json), and [release manifest](../neural/release.json) retain hashes and settings. A fresh checkout without the local weights uses the pinned published model.

The causal CWformer v4 engine remains optional and experimental: it improves synthetic results but makes more errors than RNN-T on the tiny real holdout. Its [freeze record](cwformer-freeze.json) precedes final evaluation. No model or filter tuning followed these final results.

## Frozen final evaluation

| Pipeline | Synthetic CER | False characters / 8 min | Real human CER | CW + receiver noise CER |
| --- | ---: | ---: | ---: | ---: |
| Adaptive, October 1 | 23.78% | 73 | 45.95% | 10.96% |
| Adaptive, current | 15.56% | 0 | 40.54% | 5.56% |
| Published RNN-T + original stream decoder | 18.12% | 77 | 24.32% | 14.77% |
| RNN-T v1 + current stream decoder | 15.12% | 19 | 16.22% | 6.87% |
| Published CWformer + local causal runtime | 23.55% | 59 | 64.86% | 16.23% |
| CWformer v4 + same causal runtime | 10.13% | 1 | 35.14% | 5.41% |

- **Synthetic:** 96 complete continuous messages, 3,002 reference characters, plus 24 negative clips totaling 480 seconds. Total audio is 42.30 minutes. The independent TypeScript generator covers clean, uneven human-like timing, fading, adjacent CW, rough fist, and combined chaos. These are simulated operators.
- **Real human:** two independently transcribed Marine Electric operators, 30.4 seconds and only 37 characters. Original RNN-T makes 9 errors, v1 makes 6; CWformer v4 makes 13. This sample is much too small for a broad real-world quality claim.
- **Mixed:** 24 synthetic messages mixed with actual 40 m receiver interference, 684 characters and 9.16 minutes. This is not human-sent CW. The noise recording was held out from our fine-tuning; upstream CWformer may already have trained on it.

All engines receive the known carrier, excluding automatic acquisition. RNN-T/adaptive use 100 Hz bandwidth; CWformer uses 150 Hz. Set the UI’s Filter width to 150 Hz for that CWformer configuration. CER includes spaces, punctuation, substitutions, deletions and insertions; it can exceed 100%. Noise counts exclude spaces. Zero false copy in eight minutes is not a guarantee. RNN-T results compare the whole old/new pipeline, so they do not isolate the contribution of model weights from streaming fixes.

| Synthetic condition | Original RNN-T | RNN-T v1 | Adaptive current | CWformer v4 |
| --- | ---: | ---: | ---: | ---: |
| Clean | 0.00% | 0.00% | 0.00% | 0.00% |
| Human-like timing | 1.16% | 0.78% | 0.19% | 0.00% |
| Fading | 0.76% | 0.95% | 1.14% | 0.00% |
| Nearby station | 0.00% | 0.38% | 0.00% | 0.00% |
| Rough fist | 30.77% | 18.30% | 11.14% | 9.02% |
| Combined chaos | 80.54% | 72.06% | 80.54% | 52.02% |

Severe distortion remains a major failure case. Small clean-condition regressions are visible rather than hidden by the aggregate. SNR in the synthetic generator is keyed carrier RMS versus broadband 8 kHz noise RMS before fading, not an RF measurement. Mixed-recording levels describe only the additional recorded noise, not total SNR.

Full per-clip predictions and hashes:

- RNN-T: [synthetic](rnnt-final-synthetic-selected.json), [real](rnnt-final-real-selected.json), [mixed](rnnt-final-mixed-selected.json); original reports are linked from the [comparison](rnnt-final-comparison.json).
- Adaptive: [synthetic](adaptive-continuous-final.json), [real](adaptive-real-final.json), [mixed](adaptive-receiver-noise-final.json), each with the October 1 baseline (`2949031`).
- CWformer: [synthetic](continuous-final-cwformer-v4.json), [real](real-final-cwformer-v4.json), [mixed](receiver-noise-final-cwformer-v4.json); matching `*-cwformer-published.json` reports use the same runtime.

## Reproduce

Install the Python requirements and FFmpeg as described in the [main README](../README.md). Download only the checksum-pinned corpus files; audio stays outside Git. These commands prepare the prescribed final sets, not the smaller default development set:

```sh
python3 neural/fetch_corpus.py
bun scripts/continuous-benchmark.ts --split final --cases-per-condition 16 --noise-repeats 4 --output .research/continuous-final
.venv/bin/python neural/benchmark.py --prepare neural/corpus-holdout.json --output .research/real-final/manifest.json
.venv/bin/python neural/mix_receiver_noise.py --manifest .research/continuous-final/manifest.json --output .research/receiver-noise-final/manifest.json
```

Use a fresh output directory when recreating mixed audio; the mixer refuses to overwrite existing files. With the local release weights available:

```sh
mkdir -p .research/reports
for suite in continuous-final real-final receiver-noise-final; do
  .venv/bin/python neural/benchmark.py --manifest ".research/$suite/manifest.json" --decoder original --threads 2 --output ".research/reports/$suite-original.json"
  .venv/bin/python neural/benchmark.py --manifest ".research/$suite/manifest.json" --checkpoint models/cw-boi-rnnt-v1.pt --decoder app --confidence-threshold 0.6 --digit-threshold 0.9 --threads 2 --output ".research/reports/$suite-rnnt-v1.json"
  bun scripts/benchmark-adaptive.ts --manifest ".research/$suite/manifest.json" --baseline 2949031 --output ".research/reports/$suite-adaptive.json"
done
```

The published baseline must explicitly use `--decoder original`; the default is the current app wrapper. The service’s `--published` flag selects published weights with the current wrapper. The committed [original](rnnt-final-commands-original.json) and [selected](rnnt-final-commands-selected.json) argv records show the executed commands. `models/cw-boi-rnnt-v1.pt` is byte-identical to the selected blend. Downloaded/trained weights are local artifacts, not included in Git.

For optional CWformer, repeat this command for each manifest above. The model's adjacent mel assets are required. Substitute `.research/cwformer-release/cwformer_streaming_fp32.onnx` to reproduce its published-weight baseline:

```sh
.venv/bin/python neural/benchmark_cwformer.py --manifest .research/continuous-final/manifest.json --model models/cwformer-adapt-v4/cwformer_streaming_fp32.onnx --bandwidth 150 --output .research/reports/continuous-final-cwformer.json
```

These final seeds and recordings are now consumed evaluation data. Use new reserved data for any future release selection.

## Training and decoder changes

The RTX 3080 ran bounded experiments using Python 3.12, PyTorch/torchaudio `2.8.0+cu128`, and Morseformer `0.6.4`.

- RNN-T v3 added dense six-second crops from longer transmissions, with **only fully contained characters labeled**. Partial characters at crop edges remain acoustic context. V4 added the two 20 m receiver-noise recordings as interference under labeled CW; those files are not treated as guaranteed blank audio.
- Both v3 and v4 ran 4,000 steps, batch 8, with 80% continuous examples and a 45% hard-condition fraction. V4 started from v3 step 2,000. Development selection chose a 75% blend of v4 step 4,000 and 25% effective published EMA parameters. The final confidence thresholds remain 0.6 / 0.9 for digits. Settings: [v3 run](training-continuous-v3/run.json), [v4 run](training-radio-noise-v4/run.json), [selection](rnnt-selection.json).
- The RNN-T wrapper reconciles token timestamps across overlapping windows and retains at most 200 ms of initial exact digital silence. It does not trim weak nonzero signals or internal pauses.
- CWformer training preserves CTC boundary spaces, uses complete utterances with actual silence between them, and normalizes empty-target loss by frame count. This corrects false spacing labels and disproportionate blank-example loss. V4 ran 2,400 steps; step 2,000 was selected on development data. [Run metadata](training-cwformer-v4/run.json).
- Adaptive decoding estimates local spectral noise and carrier prominence, while timing clusters account for gap frequencies and tempo changes. The original 48-case development suite improves from 9.64% to 6.43% CER; that reused suite is not final evidence.

A separate bounded RNN-T experiment can start from the published weights:

```sh
python3 neural/fetch_corpus.py --only noise
.venv/bin/python neural/train.py --device cuda --steps 600 --batch-size 8 --workers 2 --eval-every 200 --eval-count 48 --seed 660001 --continuous-fraction 0.8 --hard-fraction 0.45 --noise-recordings .research/data/cwformer/noise_20m_day.wav .research/data/cwformer/noise_20m_night.wav --save-checkpoints --output models/new-experiment
```

This is an example new run, not an exact reconstruction of the multi-stage release. Development `best.pt` gates are insufficient for promotion; freeze weights/settings before a new final evaluation. Never train on the 40 m reserved recording or the real final clips.

## External comparison and evidence limits

[Gerke 3.2.11](gerke-independent-baseline.json) was tested separately on the development corpus with its default integrating decoder, supplied carrier and nominal speed. On 18 synthetic development messages it scored 65.65% CER; on the two SAQ development crops it scored **2%**, versus **4%** for both current adaptive and selected RNN-T. Gerke's synthetic negatives produced 76.5 false characters/minute.

This is a documented configuration comparison, not an exhaustive tuning contest: Gerke decodes whole files offline, SAQ is its own upstream example, and each unknown Morse pattern counts as one replacement character. Raw output and raw scores are retained. Two correlated SAQ crops cannot establish performance across operators; the Marine holdout adds only two operators. [Corpus provenance](CORPUS.md) records transcript review, licenses, exact crop boundaries and exclusions. Larger independently annotated operator/session splits remain necessary for a best-in-class claim.

## Application verification

22 Bun tests and the production build pass. Offline checks cover complete crop labels, streaming boundaries, chunk invariance, CTC labels/loss, finite input, and bounded state. Both HTTP engines were exercised with partial audio reads, retuning, cancellation, invalid inputs, origin restrictions and session limits; the Terminal bridge uses simulated capture in integration tests. Browser sender → AudioWorklet → RNN-T v1 recovered `CQ TEST 73` exactly, as did adaptive loopback.

Actual Terminal/FFmpeg BlackHole capture succeeded after Terminal microphone permission was enabled. Its faint CW has no verified transcript, so it is not an accuracy test. A complete live hardware session through `listen.py`, physical radio reception accuracy, and hardware transmitter keying remain unvalidated.

## Historical: October 1

The earlier `rough-fist-gpu-v2` candidate was **not promoted**. On 128 independent six-second Python-generator clips it improved clean/rough/chaos CER from 8.16/24.60/67.58% to 2.04/4.81/48.90%, but regressed on clean and rough continuous TypeScript cases. [Historical holdout](final-holdout-v1.json), [published comparison](published-model-comparison.json), [rejected candidate](candidate-model-comparison.json), and [old adaptive run](adaptive-synthetic-v1.json) preserve those results. They are not the current release evaluation.
