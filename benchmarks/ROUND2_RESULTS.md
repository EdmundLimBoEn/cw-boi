# Round 2 frozen evaluation

CWformer weighted v6 step 2,000 reduces fresh synthetic errors from RNN-T v1's **449 to 299**, a 33.4% reduction, and new real-session errors from **11 to 7**. It passes every preregistered guard against both CWformer v4 and RNN-T v1. The consumed Marine recording remains worse than RNN-T: **12 versus 6 errors**.

The adaptive candidate **fails** its required synthetic improvement: total errors tie the baseline. Its stronger real-recording result does not change that decision. The candidate is archived, and deployed adaptive source and tests are restored to `87ced16`.

| Frozen pipeline | Synthetic errors / 3,027 | Synthetic CER | False characters / 8 min | New real session errors / 422 |
| --- | ---: | ---: | ---: | ---: |
| Adaptive, `87ced16` | 449 | 14.83% | 0 | 71 (16.82%) |
| Adaptive candidate, rejected | 449 | 14.83% | 0 | 48 (11.37%) |
| RNN-T v1 | 449 | 14.83% | 17 | 11 (2.61%) |
| CWformer v4 | 308 | 10.18% | 0 | 11 (2.61%) |
| CWformer weighted v6 | 299 | 9.88% | 0 | 7 (1.66%) |

The synthetic final uses previously reserved seed `2203917`: 96 messages and 24 negative clips, 42.24 minutes total. The real final contains nine correlated crops, 296.458 seconds, from one independently annotated SAQ 2022 reception. Audio and source/model hashes are recorded; all candidates were frozen before either final set was generated or unsealed. Adaptive and RNN-T use 100 Hz bandwidth; both CWformer models use 150 Hz. No word language model or dictionary rewrites the copy.

| Synthetic condition | Reference characters | Adaptive baseline → rejected candidate errors | RNN-T v1 errors | CWformer v4 → v6 errors |
| --- | ---: | ---: | ---: | ---: |
| Clean | 527 | 0 → 0 | 2 | 0 → 0 |
| Human-like timing | 525 | 0 → 0 | 4 | 0 → 0 |
| Fading | 530 | 10 → 13 | 15 | 6 → 3 |
| Nearby station | 535 | 0 → 0 | 1 | 0 → 0 |
| Rough fist | 386 | 35 → 37 | 59 | 26 → 28 |
| Combined chaos | 524 | 404 → 399 | 368 | 276 → 268 |

All seven guards are evaluated without alteration in [the full comparison](round2-final-comparison.json): strictly lower aggregate synthetic CER; no more than two percentage points of regression in each clean/human/fading/crowded condition; no increase in noise false copy; no more than one additional real-session character error. CWformer passes both its own upgrade and default-engine comparisons. Adaptive passes six guards and fails the strict aggregate one.

The consumed 37-character Marine Electric regression remains a limitation: RNN-T v1 makes 6 errors, CWformer v4 makes 13, and v6 makes 12. V6 therefore remains worse than RNN-T on that recording despite winning the newly reserved comparison. The rejected adaptive candidate improves that consumed example from 15 to 3 errors; those gains are not deployed. CWformer v6 also regresses by two characters versus v4 on the fresh rough-fist condition. Its combined-chaos CER remains **51.15%**.

These are supplied-carrier evaluations. They do not test automatic acquisition, prove universal superiority, or establish error-free recovery below the noise. The real holdout was limited to unambiguous portions of one session, with receiver noise reduction already applied. Both final sets are now consumed and must not be reused to select further changes.

The RTX 3080 ran two bounded 3,000-step experiments. Causal-front-end retraining regressed real development copy and was rejected ([comparison](training-cwformer-causal-v5/comparison.json)). The selected run starts from original CWformer v4 and shortens only intra-character gaps to 0.25–1 times normal on 40% of non-noise utterances, preserving letter/word spacing and complete labels. Step 2,000 was selected on development results before final evaluation: 99 → 16 real errors/438 characters, unchanged 148 synthetic errors/1,467, and noise 1 → 0/4 minutes. This is one training seed without a matched unweighted continuation, so augmentation alone is not proven to cause every gain. [Run settings](training-cwformer-weighted-v6/run.json), [selection](training-cwformer-weighted-v6/comparison.json), and [artifact freeze](training-cwformer-weighted-v6/freeze.json) retain the details. A separate proposed pause-reset change was rejected after an independent 34-case check found a regression ([results](round2-cwformer-pause-check.json)).

[Plan and provenance](round2-plan.json), [exact evaluation commands](round2-final-commands.json), [corpus reproduction](CORPUS.md), and [all per-clip reports](round2-final-comparison.json) preserve the evidence. No model or threshold tuning followed these final results. [Deployment](round2-deployment.json) changes only the selected model path and label; the running service explicitly uses CWformer v6, with RNN-T available through the browser selector.

The separate published encse LSTM baseline scored 19.42% CER and 10 false characters/minute on the consumed small synthetic development set with a supplied-carrier filter ([protocol and results](continuous-dev-encse-published-bandpass.json)); the old two SAQ development crops decoded exactly ([results](real-dev-encse-published-bandpass.json)). This limited configuration comparison was not used for final promotion, and that model is not integrated into the app.

Deployment checks pass: 23 Bun tests, production build, Python generator/streaming/corpus checks, and real HTTP integration for both neural engines. In the browser, CWformer v6 recovers the 39-character human-timing lab message and sender → AudioWorklet → neural `CQ TEST 73` loopback exactly. Actual live BlackHole reception still has no verified transcript.
