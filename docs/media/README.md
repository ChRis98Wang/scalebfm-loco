# Published demonstration media

[Open the complete gallery](../DEMO_GALLERY.md) ·
[Inline GIFs in the repository README](../../README.md) ·
[Source credits and boundaries](../MEDIA_SOURCES.md)

The 2026-09-10 gallery contains **13 GIFs, 26 MP4s and 13 PNG posters**:
6 current native IsaacLab recordings, 2 MuJoCo sparse-target controls,
3 earlier MuJoCo body-motion comparisons and 2 explicitly failed task trials.

Each prefix has a `.gif`, captioned `.mp4`, unchanged `-raw.mp4` and poster
`.png`. All media passed full decoding / expected-frame-count checks; original
recordings and existing captions are preserved. Full-window GIFs play at 1x.
This is sampled visual review plus full automatic decode, not complete manual
playback or a new physical acceptance run.

The machine-readable [gallery catalog](gallery-20260910.json) lists every file,
SHA-256, byte count, source report hash, duration and outcome. Total media size is
**78,158,772 bytes**. Existing per-recording archive JSONs are historical receipts:
their local-only/publication/rights flags are not rewritten by this later push.

All demos use official pretrained weights, not new locally trained manipulation
policies. Keep the **failed** pushing/lifting clips and the **empty-handed**
reaching clip labelled accordingly. Do not infer complete ScaleBFM reproduction,
walking carry, generalization or hardware validation from playback.
