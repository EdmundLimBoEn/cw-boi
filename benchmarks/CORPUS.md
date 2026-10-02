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
