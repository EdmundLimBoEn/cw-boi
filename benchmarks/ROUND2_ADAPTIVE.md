# Adaptive receiver: partial keying and heavy key weighting

**Rejected; not deployed.** On the reserved final set, synthetic errors tied the baseline at 449/3,027, failing the preregistered requirement for strict improvement. The real-session improvement from 71 to 48/422 does not waive that guard. Deployed adaptive source and tests were restored to `87ced16`. The exact experiment is preserved in [adaptive-round2.patch](adaptive-round2.patch); reproduce by applying that patch to an isolated checkout of `87ced16` and running the recorded [final evaluator commands](round2-final-commands.json). [Final decision and evidence](ROUND2_RESULTS.md).

Baseline: `87ced16`. Candidate DSP SHA256: `ab3fc739685bc76592acb579f7a6debff493f11910d6c33aa61ac905d81466db`. `src/timing.ts` unchanged. These are development and consumed regression datasets, not the reserved round-two final holdout.

Two independent failure mechanisms were reproduced:

- Receiver AGC amplifies a residual tone during key-up. The decaying peak gate retriggers and shortens or erases gaps. A bounded one-second amplitude history now estimates two levels. This gate only engages when the lower level exceeds local noise, the upper level is at least twice as large, and the lower cluster has coefficient of variation below 0.6. Without these guards, noise, fading, and modest carrier flutter regressed.
- Heavy key weighting violates the assumption that a dit and its following gap have equal duration. In GPK development audio, approximately 90 ms dits and 210 ms dahs accompany 27 ms element gaps. The old gap prior picked a 38 ms dit and misread almost every dot as a dash. Marks now dominate initial speed estimation; existing independent mark/gap timing adaptation remains intact.

| Corpus | Reference characters | Baseline errors | Candidate errors | Noise false characters |
| --- | ---: | ---: | ---: | ---: |
| round2-dev | 1467 | 229 | 227 | 1 → 1 |
| round2-real-dev | 438 | 193 | 38 | 0 → 0 |
| round2-consumed | 3002 | 467 | 466 | 0 → 0 |
| round2-consumed-real | 37 | 15 | 3 | 0 → 0 |
| round2-supplementary | 1284 | 208 | 213 | 0 → 0 |

Full predictions, source hashes, manifest hashes, condition scores, and baseline results are in the corresponding `*-adaptive-candidate.json` reports. The synthetic error totals are essentially unchanged; the 60-case supplementary suite regresses by five characters (0.39 percentage points). Old eight-minute noise tests remain empty; new four-minute negatives retain the same single false character as baseline.

Independent physical checks are in `src/decoder.test.ts`: 50% dit-length mark extension with corresponding shortened gaps at 12/20/30 WPM; 10% same-frequency residual carrier plus 20 dB white noise and 30/100/300 ms receiver AGC release; and a continuous carrier with 30% sinusoidal amplitude flutter at 0.4/2/10 Hz. All three heavily weighted messages now decode exactly. Across four residual-carrier settings including no AGC, errors fall from 20 to 2, with one extra character in each of two cases. The flutter tests remain empty. Severe 25–40% residual carrier with fast AGC remains difficult; no claim of arbitrary modulation recovery is warranted.

Validation: `bun test src/decoder.test.ts src/timing.test.ts` passes all 23 tests. No vocabulary or callsign correction, per-recording thresholds, Python runtime changes, or new final evaluation were used.
