# Native box-lift publication to main — 2026-09-10

User-requested destination: `ChRis98Wang/scalebfm-loco`, branch `main`.
The existing official `zengweishuai/ScaleBFM` remote is not a publication target.

The publication continues the destination's initial commit `029659c` through the
already published source snapshot `512a1a1`. It uses an isolated publication
worktree, a normal fast-forward push, and no force update. The local development
branch, its uncommitted changes and the previous publication worktree are retained.
The Git commit containing this page is the publication identifier.

## Included in this update

- A directly embedded, original-speed **box-lift GIF** in the root README.
- Captioned MP4, unmodified raw MP4, PNG and the original archive JSON under
  `docs/media/isaac-lift-single-state-20260910*`.
- The native lift runner, planner implementations, fixed experiment configs,
  independent physical-sequence audit, 57 logic tests and technical documentation.
- Explicit explanation that the planner uses simulator state through IsaacLab
  APIs, while the BFM consumes proprioception and sparse targets, **not vision**.

No motion arrays, AMASS/SMPL-X inputs, robot meshes, checkpoints, SDKs, credentials
or full private experiment logs are included. KIT/ACCAD-derived dance and motion
GIFs remain outside this publication pending their separate review. Historical
protocols and the archive JSON are preserved byte-for-byte, including statements
that no publication occurred during those earlier experiments.

## Publication checks

- Existing explicit CPU subset: **138 passed** in the isolated source checkout.
- New controller tests: **49 passed**; independent acceptance logic: **8 passed**.
- Published runtime source/protocol/config files: **54 recorded hashes matched**.
- Media: all four SHA-256 / size checks passed, totaling **16,179,655 bytes**.
- Full decode: raw and captioned MP4s **971 frames each**, GIF **194 frames**, 1x.
- README, demo report, media README, snapshot and notices: **69 local links valid**.

All checks are local. The published video was generated in the previous physical
trial, not re-simulated for the push. No hosted CI success is claimed here.

The box task passed one development initial state with the official pretrained
policy, not a newly trained manipulation model. Multi-initial-state D1 acceptance,
walking carry, visual perception and hardware contact calibration are not complete.
See the [physical result and API/control explanation](ISAAC_BOX_LIFT_DEMO_20260910.md)
and the [source/asset boundary](SOURCE_SNAPSHOT.md).
