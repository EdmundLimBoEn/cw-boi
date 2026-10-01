# Third-party software

The optional neural engine uses **Morseformer 0.6.4** by Sébastien Derhy, licensed under Apache License 2.0.

- Source: https://github.com/sderhy/morseformer
- Weights/model card: https://huggingface.co/sderhy/morseformer
- Pinned weights: `rnnt_phase11b.pt`, revision `9eab86a3ad7482f8c5801eabf26ec46f9493b919`
- License: https://github.com/sderhy/morseformer/blob/main/LICENSE

CW boi's Python service wraps the published inference implementation; its fine-tuning script adds a separate distortion generator, evaluation gates, and training loop. No optional vocabulary/language model is used. Model weights are downloaded separately and excluded from Git. Fine-tuned weights remain derivatives of the published checkpoint and retain its license.

The interface uses React, Vite, Lucide icons, Space Grotesk, and IBM Plex Mono. Their licenses are distributed with their installed packages.
