# Public recording corpus

Fetch pinned originals with `python3 neural/fetch_corpus.py`. Downloads stay in ignored `.research/data`; URLs, sizes, SHA-256 checksums, licenses and reviewed crop labels are in `neural/corpus.json`. FFmpeg is required to derive the two Marine Electric WAVs from the downloaded MP3 prefix. `python3 neural/fetch_corpus.py --check` verifies checksum failure preserves an existing file.

## Real receiver noise

[CWformer](https://github.com/parsimo2010/CWformer/tree/ef6ac7ca75b20833c811ea9ebf2bde1fa139fa70/recordings) provides three actual HF noise recordings under the repository's MIT license. Its license is downloaded alongside them.

| Recording | Duration | Split |
| --- | ---: | --- |
| 20 m, day | 732.192 s | Training augmentation |
| 20 m, night | 666.816 s | Training augmentation |
| 40 m, day | 649.920 s | Held-out receiver condition |

These are additive interference sources. The original author calls them band noise; we have not verified every second is free of weak Morse. Do not assign the entire files empty-transcript ground truth. The held-out recording must stay out of training. CWformer itself may already have trained on these files, so they cannot establish unseen-noise generalization for that upstream model.

## Reviewed SAQ reception

[Gerke Decoder's author](https://github.com/fowlay/gerke-decoder#test-data) supplies a 94.53-second receiver recording of the June 30, 2019 morning SAQ transmission. The [operator's event report and message](https://radiomuseet.se/medlem/nyhetsbrev/nyhetsbrev_127.pdf#page=8) corroborate the website text. The [official reception report](https://alexander.n.se/wp-content/uploads/2019/07/Preleminary-Summary-Report-from-Alexanderson-Day-June-30-2019.pdf) describes sender corrections and “cw handwriting”; this is the live message, not the automated pre-transmission identification loop. The specific key model and operator identity have not been independently established for these samples.

| Original time range | Reference |
| --- | --- |
| 1.10–25.20 s | `OUR WEBSITE : WWW.ALEXANDER.N.SE` |
| 26.65–41.50 s | `WWW.ALEXANDER.N.SE` |

The actual mark/gap sequence was checked independently of either neural model against the published text. Three fixed envelope thresholds agreed on the characters. The colon has long gaps on both sides in the recording, hence the spaces in the reference. Crops include complete characters and avoid the preceding/following operator signals. Pitch is approximately 634 Hz.

These two clips are **development data from one operator and one recording**, with a repeated phrase. They do not constitute a diverse human-sent benchmark. The original audio has no explicit reuse license, so it remains a local evaluation download and is excluded from model training and redistribution.

Prepare the benchmark files with:

```sh
.venv/bin/python neural/benchmark.py --prepare neural/corpus.json --output .research/real-curated/manifest.json
```

## Independent Marine Electric check

[SS Marine Electric / WOOH SOS](https://archive.org/details/SsMarineElectricWoohSos) is actual February 12, 1983 US Coast Guard reception, digitized by N1EA. [Archive metadata](https://archive.org/metadata/SsMarineElectricWoohSos) declares CC0 and supplies a transcript. That transcript contains commentary, uncertain passages and non-audio explanations; it must not be used verbatim as the reference for the full recording.

| Original time range | Operator | Pitch | Reference |
| --- | --- | ---: | --- |
| 173.700–183.350 s | NMF / USCG COMMSTA Boston | 708 Hz | `DE NMF NMF ? K` |
| 84.550–105.300 s | WOOH / SS Marine Electric | 790 Hz | `DOWN BY HEAD AT 0800GMT` |

The source's unspaced `IMI` prosign is normalized to `?` because both are `..--..`. NMF's first complete D starts at 173.889 s and the last K ends at 182.902 s. Each actual Morse pattern and character boundary is recorded in `neural/corpus-holdout.json`. WOOH's first D starts at 84.819 s and last T ends at 104.562 s; independent envelope review at three thresholds gives the identical publisher-transcribed text.

These 30.4 seconds are kept separate from development and training. They cover two additional operators, but remain a tiny sample of human keying. Initial annotation used the publisher transcript plus direct mark/gap review, without either neural model's predictions.

The acquisition stores the first 1,048,576 bytes of the MP3 with an explicit HTTP Range request and a pinned SHA-256 checksum. This intentional prefix contains 222 seconds, enough for both clips. Its bytes agree across independent downloads. The full 91-minute original was **not** acquired or locally hash-verified. A bounded attempt was stopped; no download processes remain.

Reproduce the local crops and prepare them for the final check:

```sh
python3 neural/fetch_corpus.py --only marine-electric
.venv/bin/python neural/benchmark.py --prepare neural/corpus-holdout.json --output .research/real-holdout/manifest.json
```

Both WAV files were reproduced byte-for-byte from the pinned prefix. The fetcher verifies their hashes and fails if another decoder/resampler version changes the waveform. Do not run these final clips repeatedly to select model or filter settings.

## Rejected scored source

- [Priyom's CW Saturday](https://priyom.org/blog/cw-saturday): actual naval CW reception with a human-published transcript, CC BY-NC-SA 4.0. The downloaded recording contains extra repeated groups and a callsign that does not match the Latin transcription exactly. It is **not accepted as scored ground truth**. Its very regular element durations also do not establish human keying.

No decoder output is used as a reference transcript. Original recordings, conversions and exploratory predictions are not committed.

## Round 2: expanded real development and a new session holdout

The earlier SAQ 2019 and two Marine Electric checks above have now been evaluated. They remain **consumed regression examples**, not fresh final tests. Their original manifests and reports are unchanged.

`neural/corpus-dev-v2.json` adds **12 non-overlapping clips, 438 reference characters including spaces, and 290.87 seconds**. These cover three operators and two recording sessions:

| Session | Operators | Clips | Reference characters |
| --- | --- | ---: | ---: |
| Marine Electric distress traffic, 1983 | WOOH and LJKR | 8 | 237 |
| Portpatrick Radio final broadcast, 1997 | Graham Mercer / GM4BES, GPK | 4 | 201 |

Marine Electric labels use the independent [published human transcript](https://archive.org/download/SsMarineElectricWoohSos/Transcript_SOS_WOOH.txt). The actual dits, dahs, word gaps and crop boundaries were checked with narrow-band envelope extraction at three or four fixed thresholds; all retained characters agree. The reference preserves transmitted number groups and spacing, including a question-mark prosign attached directly to its preceding group. These examples share the source session used by the old tiny regression check, so they do not establish session-independent generalization.

The [GPK recording](https://zl1.nz/about-amateur-radio/new-zealand-nets/nz-net/newsletters/nr106/) is corroborated by [Chris G3TUX's contemporary first-person transcription](https://www.jproc.ca/radiostor/fwcwgb.html), which describes the closing transmissions as hand-sent. Independent I/Q envelope inspection at three thresholds verifies every retained Morse pattern and gap. Actual repeated tokens and attached prosigns are retained; editorial spacing is not imposed on the signal. Ambiguous or differently transcribed regions were excluded before evaluating the models. These four crops are one operator, not four independent tests.

The new development manifest contains source URLs, byte counts, source and crop SHA-256 hashes, original time ranges, carrier settings, and per-character mark/gap evidence. Neither app output nor neural predictions supplied any reference label. **All v2 real speech remains evaluation-only and is excluded from training.** Marine Electric has the uploader's CC0 declaration. No open redistribution license was found for the GPK recording, so it remains a local evaluation download; neither original nor derived audio is committed.

The earlier acquisition had completed the full Marine Electric MP3, whose MD5 matches Archive metadata. Reproduction of v2 needs only its pinned first 9 MiB, sufficient for the last reviewed crop at 1599 seconds. It does not redownload the full 91-minute recording.

```sh
.venv/bin/python neural/fetch_corpus_v2.py --check
.venv/bin/python neural/fetch_corpus_v2.py
.venv/bin/python neural/benchmark.py --prepare neural/corpus-dev-v2.json --output .research/real-dev-v2/manifest.json
```

All 12 WAV crops were re-derived using the fetch helper and reproduced with identical SHA-256 hashes. Conversion failure leaves any existing valid file intact.

### Round 2 real test, now consumed

`neural/corpus-final-v2-sealed.json` preserves the original seal: **nine non-overlapping crops, 422 reference characters, 296.458 seconds**, from the [November 16, 2022 SAQ reception by Richard B. Langley](https://shortwavearchive.com/archive/saq-grimeton-radio-november-16-2022). The entire 570.63-second session was excluded from development and training. After all candidate configurations were frozen, the annotation was published unchanged as `neural/corpus-final-v2.json`, SHA-256 `4ff69881fd40e26a9c6649c05ea4176bf8f898009418101e6411f12fc862a053`. Audio remains local and outside Git.

The receiver-authored transcript was independently checked against four fixed envelope thresholds. Only complete-word regions with agreement were retained, before any model evaluation; unclear passages and disagreements were excluded rather than corrected using a model. The site declares CC BY-NC 3.0. Attribution is retained and the recording is used only for local evaluation, with no training or redistributed audio.

This is one transmitter/receiver session, with correlated crops and receiver noise reduction already applied. It measures generalization to an additional session; it does not establish performance for every form of human keying or severe interference. Unknown exposure during upstream pretrained-model development cannot be excluded. The [one-time final comparison](round2-final-comparison.json) is complete. This session and synthetic final seed `2203917` are now **consumed evaluation data**; no candidate was tuned after their results.

```sh
.venv/bin/python neural/fetch_corpus_v2.py --manifest neural/corpus-final-v2.json
.venv/bin/python neural/benchmark.py --prepare neural/corpus-final-v2.json --output .research/round2-real-final/manifest.json
```

The prepared 8 kHz corpus manifest has SHA-256 `89abd7407877fefd17651de4e80a1a7fcf8ac20b4655815bb20f5f5e53d28caf`. Its hash differs from the source annotation because it includes conversion metadata and references the derived float32 audio files. Actual evaluator commands, source and model hashes, original release guards, and all predictions are retained in [the round 2 plan](round2-plan.json), [commands](round2-final-commands.json), and comparison reports.

## Round 3 reserved human session

[Sealed metadata](../neural/corpus-final-v3-sealed.json) reserves five crops, 237 reference characters and 97.662 seconds from [NW6V's straight-key recording](https://archive.org/details/sending_morse_code_sound_recordings). Three fixed envelope thresholds agree on the selected marks and character boundaries; the sender's published text supplies word spacing. Ambiguous boundaries and a sender-error region were excluded before model evaluation. The private annotation has SHA-256 `0f52be1369e8b3777e4c0a80257bd0fca787dd8ce7808aa70a0c0bc7c262bd27`.

This is one new operator/session, with relatively clean local sidetone. It tests human timing, not weak RF reception. The uploader declares CC BY-NC-ND 4.0; audio is kept local, with no redistribution or training use. The entire session remains reserved until a model candidate is frozen and admitted to final evaluation under the [round 3 plan](round3-plan.json).
