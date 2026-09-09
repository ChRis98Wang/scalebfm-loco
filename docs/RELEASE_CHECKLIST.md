# Public-release preparation

Status: source-only engineering snapshot prepared for a user-authorized push to
`ChRis98Wang/scalebfm-loco`, on `publish/scalebfm-20260909`. The snapshot is based
on that repository's initial README commit, not on the asset-bearing local Git
history. No data, weights, robot assets, SDK binaries, logs or videos are included.
See [the publication boundary](SOURCE_SNAPSHOT.md). This checklist is an
engineering audit, not a grant of redistribution rights or a completed release.

## Completed safeguards

- Root README labels this as an unofficial fork and preserves upstream attribution.
- Root ignore rules exclude local logs, common checkpoint files, AMASS/SMPL-X
  working directories, partial downloads, caches and credential files.
- Native GUI entry point supports `--help`, `--dry-run`, local-file preflight,
  a fixed duplicate-resistant transient unit and a 30-minute process lifetime.
- Test instructions distinguish CPU checks, licensed-data work and real simulator checks.
- Simulator-independent CI is configured with Python 3.11/3.12, read-only permissions,
  pinned actions, a job timeout and no artifact uploads; the test subset passed locally
  in an isolated Python 3.11 environment without IsaacLab, data or weights.
- The optional target-object scene documents uncalibrated physical parameters and
  separates object diagnostics from learned manipulation or task-success claims.
- Independent official-vs-fine-tuned evaluation completed with an explicit protocol,
  unchanged-input manifests and per-clip results; the regression is disclosed in
  [the local evaluation report](BFM_EVALUATION_20260907.md), not hidden by a reproduction percentage.

Ignore rules do **not** remove files already tracked by Git and are not a secrets scanner.
Never publish by blindly staging the entire working tree.

## Required before a public release

- [ ] Resolve upstream root licensing. This checkout has no root LICENSE; package
  metadata saying MIT is not enough to relicense every component.
- [ ] Inventory and preserve the licenses of bundled RSL-RL, IsaacLab-derived code,
  Unitree assets/SDK and other third-party code. Some bundled code has BSD-3-Clause headers.
- [ ] Review already tracked example motions in `ScaleTrack/source/scaletrack/data/example/`
  and `ScaleBridge/scalebridge/data/motion/` before any future asset release;
  these local historical files and their Git ancestry are excluded from this snapshot.
- [ ] Independently verify permissions for raw motions, retargeted motions, SMPL-X
  models, pretrained weights and any fine-tuned weights before distributing them.
  Default release scope is source, tests and documentation only.
- [ ] Separate historical local experiment notes (absolute paths and local file links)
  from portable public setup instructions; audit tracked and staged files for secrets.
- [ ] Capture exact local IsaacLab revision/patches and dependency versions, and
  test a clean editable installation before advertising reproducibility.
- [ ] Verify wheel package discovery/assets; current editable installs do not prove wheels work.
- [ ] Run and verify the configured CI on GitHub after an authorized source-only push;
  local checks do not establish hosted-run success. Keep licensed/GPU checks separate.
- [ ] Prepare a portable, rights-reviewed public evaluation summary from the local
  result. No local logs, data, weights or private paths should be uploaded blindly.
- [x] User selected `ChRis98Wang/scalebfm-loco` and authorized the source push.
- [ ] Obtain final release approval after the remaining licensing and runtime checks.

Useful read-only checks:

```bash
git status --short
git diff --check
git diff --cached --name-only
git ls-files '*.npz' '*.pkl' '*.pt' '*.pth' '*.ckpt' '*cookie*' '.env*'
```
