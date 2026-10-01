# Measured results, 2026-10-01

The default application retains the published Morseformer checkpoint and the browser adaptive decoder. **The fine-tuned candidate was not promoted**: it improved an independent synthetic holdout but regressed on two cases from the separate TypeScript generator used by the streaming service.

## Independent holdout

128 complete six-second clips, seed `3000000`, 32 clips in each group. Training and checkpoint selection did not use this seed. Both checkpoints were evaluated on CPU with the same acoustic confidence thresholds. Character error includes spaces and insertions; it is not a word-accuracy percentage.

| Condition | Published model | GPU candidate |
| --- | ---: | ---: |
| Clean, 147 reference characters | 8.16% CER | 2.04% CER |
| Rough timing, 187 characters | 24.60% CER | 4.81% CER |
| Combined distortions, 182 characters | 67.58% CER | 48.90% CER |
| Noise-only, 192 seconds total | 50 false characters | 0 false characters |

Full predictions and the checkpoint SHA256 are in [final-holdout-v1.json](final-holdout-v1.json). These are new seeds from the Python generator family used in training, not real radio recordings. Zero false characters here does not establish a zero false-positive rate.

## Independent generator and streaming

The TypeScript generator uses seed `39181` and a known carrier. Its longer messages go through the same six-second sliding-window decoder used by the app.

| Case | Published CER | Candidate CER |
| --- | ---: | ---: |
| Clean | 0.00% | 2.56% |
| Human timing | 2.56% | 0.00% |
| Fading | 0.00% | 0.00% |
| Nearby station | 0.00% | 0.00% |
| Very rough fist | 10.26% | 15.38% |
| Combined chaos | 79.49% | 71.79% |

The clean regression is a missing word space. The rough-fist regression includes missed characters and spacing. A six-second full-clip validation improvement is insufficient evidence that continuous decoding improved. See [published-model-comparison.json](published-model-comparison.json) and [candidate-model-comparison.json](candidate-model-comparison.json).

## Runs and selection

- CPU pilot: 600 steps, batch 4, seed `110001`; no accepted checkpoint.
- RTX 3080 first run: 600 steps, batch 4, seed `220001`, 45% combined-distortion samples; 62 seconds after baseline evaluation. Hard-case CER tied the published checkpoint, so no promotion.
- RTX 3080 continuation: 4,000 steps, batch 8, seed `330001`, 45% combined distortions, punctuation included; 295 seconds after baseline evaluation. Initialized from the first GPU run. Checkpoint selection used 48 other clips, seed `2000000`; step 3,000 was best among checkpoints passing that run's development gates. The last checkpoint did not pass those gates.

The selected checkpoint is `models/rough-fist-gpu-v2/best.pt` locally and `C:\Users\limbo\cw-boi-training-20261001\models\rough-fist-gpu-v2\best.pt` on chonkus. Weights are excluded from Git. Reports in the `training-*` directories retain individual validation examples, seeds, versions, and run settings. The GPU jobs have completed.

The adaptive decoder separately achieved 32/48 exact messages and 9.64% aggregate CER on the fixed random-message suite in [adaptive-synthetic-v1.json](adaptive-synthetic-v1.json). That suite uses automatic carrier acquisition and timing; it is not directly comparable with the neural holdout.

## Application verification

16 Bun tests and the production build pass. Python checks cover complete labels, reproducibility, mixed silence batches, and input boundaries. Live API checks reject invalid floats/settings, nonlocal origins/hosts, and excess sessions. Browser checks exercised real AudioWorklet loopback, imported 925 Hz audio with automatic tuning, keyboard straight-key `SOS`, Morse-text conversion, signal-lab scoring, and desktop/mobile layout. Physical radio/microphone reception and hardware transmitter keying were not validated.

Next evidence needed: independently transcribed real hand-keyed recordings, split by operator, receiver, and session, plus long continuous synthetic cases for future checkpoint selection. Do not select checkpoints using this final holdout.
