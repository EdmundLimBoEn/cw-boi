# Third-party software and recordings

The RNN-T engine uses **Morseformer 0.6.4** by Sébastien Derhy, under Apache License 2.0.

- [Source](https://github.com/sderhy/morseformer) and [weights/model card](https://huggingface.co/sderhy/morseformer).
- Pinned published weights: `rnnt_phase11b.pt`, revision `9eab86a3ad7482f8c5801eabf26ec46f9493b919`.
- [License copy](neural/MORSEFORMER_LICENSE).

CW boi adds a streaming token-boundary/startup-silence wrapper, independent distortion generator, evaluation harness, and fine-tuning loop. RNN-T v1 blends fine-tuned parameters with the published model's effective EMA parameters. These weights remain derivatives under the same license. No optional word language model is used. Model weights are downloaded or trained separately and excluded from Git.

The optional causal engine uses **CWformer** by parsimo2010, under the MIT license. Its separately downloaded ONNX weights and mel assets derive from v0.2.0 and local fine-tuning. The inference wrapper follows the upstream streaming frontend/cache layout, with carrier translation and filtering.

- [Source](https://github.com/parsimo2010/CWformer), pinned revision `ef6ac7ca75b20833c811ea9ebf2bde1fa139fa70`.
- [v0.2.0 release](https://github.com/parsimo2010/CWformer/releases/tag/v0.2.0).
- [License copy](neural/CWFORMER_LICENSE).

**Gerke Decoder 3.2.11** by fowlay is used only as an external benchmark executable. Its source is not incorporated into the application. The comparison uses revision `9ecceceeb04a2bae2fc563a3da79ba517194283f`, licensed GPL-3.0-or-later. [Source/license](https://github.com/fowlay/gerke-decoder) and [comparison provenance](benchmarks/gerke-independent-baseline.json).

**encse/morse-decoder** is also an external development benchmark only, under MIT, pinned to `b4244d0b2635652eaefb21a7f421e96889636e73`. Its code and weights are not incorporated into the application. [Source/license](https://github.com/encse/morse-decoder) and [comparison provenance](benchmarks/continuous-dev-encse-published-bandpass.json).

The interface uses React, Vite, Lucide icons, Space Grotesk, and IBM Plex Mono. Their licenses are distributed with their installed packages.

Recordings retain their individual terms: CWformer receiver noise is MIT; Archive.org declares the Marine Electric recording CC0; the SAQ 2022 reception page declares CC BY-NC 3.0. The reserved NW6V straight-key recording has the Archive.org uploader's CC BY-NC-ND 4.0 declaration. No open redistribution license was found for the SAQ 2019 or GPK recordings. All real CW recordings are restricted here to local evaluation, excluded from training and audio redistribution. All audio stays outside Git. [Corpus documentation](benchmarks/CORPUS.md) and the linked download manifests record URLs, hashes, licenses, transcript provenance and split restrictions.
