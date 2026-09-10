# Source snapshot and demo gallery — 2026-09-10

This is a development source snapshot, not a complete ScaleBFM reproduction,
rights-cleared release, or standalone simulator installation.

## Provenance and contents

- Destination: `ChRis98Wang/scalebfm-loco`, branch `main`, as requested by the user.
- History: the destination's initial README commit
  `029659cebeb751a55f812153df8b0b9ac8eaf343`, followed by the previously published
  source snapshot `512a1a1c2a0b6f71db3c1b669cd702dace2978eb`, then this scoped demo
  update at `8367d52`, followed by the owner's requested full playable gallery
  and shortened README. Publication is a fast-forward, not a force push or an import of the
  asset-bearing local development history. The old publication branch is retained.
- Upstream source baseline: [ScaleBFM at abd6f17](https://github.com/zengweishuai/ScaleBFM/tree/abd6f17c02fe0baabc14709feb8d9ea4959aa621).
- Local engineering base: `897b1460146def06c705253dfee7c48db509c530`, plus the
  audited working-tree changes through the Stage II v3 experiment.
- Included: main Python modules, shell entry points, configuration, packaging
  metadata, tests, CI and engineering reports, with existing author/license headers.
  The lift update adds the native lift controller/runner, fixed experiment configs,
  independent physical checks, 57 logic tests and the selected rendered media.
- Excluded: the local asset-bearing Git history, external UMR checkout, Unitree
  SDK, robot models/meshes, bundled wheels, examples/motion arrays, AMASS/SMPL-X
  inputs, checkpoints, logs, caches, credentials and all other media.

## Current media gallery

The owner subsequently requested every usable GIF/video and a concise AMASS
section. The [gallery](DEMO_GALLERY.md) now includes 13 complete recordings:
6 current native IsaacLab clips, 2 synthetic MuJoCo controls, 3 older body-motion
comparisons and 2 explicitly failed trials. All 13 GIFs and 26 MP4s are linked
from README; failed tasks remain in a separate collapsed section.

The 52 media files total 78,158,772 bytes. Full decode checks passed for every
GIF/MP4 and source hashes were verified. Only three missing GIFs were generated;
existing MP4s were not cut or re-encoded. No new policy/physics trial, training,
runtime-source changes, dataset or checkpoint publication occurred in this gallery
update. [Hashes and file inventory](media/gallery-20260910.json) ·
[Source credits and publication boundaries](MEDIA_SOURCES.md).

The full AMASS training/retargeting experiment history remains in the existing
documents; README now provides only the pipeline, data scale, model, training
conclusion and GUI/setup links. Previous local archive flags are historical,
not a new legal-clearance certificate.

## Earlier native box-lift addition

The [box-lift report](ISAAC_BOX_LIFT_DEMO_20260910.md) explains the exact split
between the official learned controller and our object-feedback sparse-target
planner. The policy has no camera input; IsaacLab exposes simulator ground-truth
state to the planner, not to a visual perception system. The single-state trial
passed; formal multi-initial-state acceptance and walking carry are still pending.

The earlier `8367d52` update added only `docs/media/isaac-lift-single-state-20260910` with `.gif`, `.mp4`, `-raw.mp4`,
`.png` and `.json` is added. The four media files total 16,179,655 bytes, copied
byte-for-byte from the audited local archive. No AMASS motion frames were used
for this box trial. The original archive receipt retains `publication_performed:
false` because it records the earlier local archival operation; this document
and Git history describe the later publication. Hashes/protocols are historical
evidence, not downloadable full simulation logs or a blanket media/asset license.

The native runtime source dependencies already present in the source snapshot
match the recorded trial's input hashes. The added controllers, runner and
configs also preserve the tested source bytes. The runner is provenance-bound:
trusted official weights, robot assets, a matching FK seed and metadata are
required separately. A fresh asset-free clone cannot launch the recorded trial
until these inputs and the existing compatible IsaacLab environment are supplied.
No dependency installer, GPU job or newly trained weights are part of this push.

UMR v1/v2/v3 experiment scripts and frozen protocols retain their local file
contents. Their reports reference local manifests and licensed inputs that are
not published here. They are provenance-bound research runners, not portable
one-command reproductions; missing inputs must fail rather than bypass checks.

## Runtime assets are separate

The included upstream module READMEs describe the complete upstream checkout.
Their statements about bundled assets/examples do **not** apply to this snapshot.
Obtain assets directly from the original source under their applicable terms.
Do not download arbitrary model files or unpickle untrusted motion archives.

For an independent, fresh workspace, first obtain the fixed upstream checkout:

```bash
git clone --no-checkout https://github.com/zengweishuai/ScaleBFM.git /path/to/ScaleBFM-upstream
git -C /path/to/ScaleBFM-upstream checkout --detach abd6f17c02fe0baabc14709feb8d9ea4959aa621
git clone --branch main https://github.com/ChRis98Wang/scalebfm-loco.git /path/to/scalebfm-loco
```

Use new destinations; do not reset or overwrite an existing development checkout.
Once the applicable asset terms are reviewed, restore only the required paths
from the pinned upstream checkout to the same relative paths in the new source
checkout. For these destinations, confirm they do not already exist before copying:

| Runtime | Missing path to obtain from upstream |
| --- | --- |
| ScaleRetarget robot geometry | `ScaleRetarget/assets/unitree_g1/` |
| ScaleTrack robot geometry | `ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof/` |
| ScaleBridge robot geometry | `ScaleBridge/scalebridge/data/robot/g1_29dof/` |
| Optional real-robot SDK | `ScaleBridge/third_party/` |

Keep the accompanying license/notice files with restored assets. SDK and
real-robot setup are optional and are not needed for the simulator-independent
tests. Refer to upstream for any LFS-managed assets; a pointer file is not a mesh.
Example motions are deliberately not a setup requirement. Supply personally
obtained motion/body-model inputs and trusted checkpoint/index paths separately.

Reuse the already-working IsaacLab and retargeting environments. This publication
does not reinstall dependencies or validate a new simulator stack. After assets,
checkpoints and motion indexes are available, the GUI entry point is:

```bash
cd /path/to/scalebfm-loco
export BFM_ISAAC_PYTHON=/path/to/existing/isaaclab/bin/python
bash scripts/open_motion_browser.sh --help
bash scripts/open_motion_browser.sh --dry-run
```

Read [the GUI guide](AMASS_MOTION_BROWSER.md) before actual playback. The launcher's
development-machine defaults are not bundled files and need overrides elsewhere.

## Evidence and limitations

The original development checkout passed 1,185 local regression tests across
its existing environments. These are local results, not GitHub-hosted CI results
or proof that the asset-free snapshot passes every simulator integration test.
The source-only checkout independently passed the three configured CPU groups
(138 tests plus 13 subtests), using existing Python 3.12.3, pytest 9.1.1 and
PyYAML 6.0.3, with no dependency installs. The bounded test service exited with
no child processes. This does not verify the hosted Python 3.11/3.12 matrix or
its locked pytest 8.4.2 dependency.

The public CI runs only the explicit simulator-independent subset in
`.github/workflows/cpu-tests.yml`; full physics and licensed-data validation remain
separate. See [testing](../CONTRIBUTING.md).

Before the 2026-09-10 `main` update, the isolated publication checkout reran the
existing CPU subset: **138 tests passed**. Its new lift tests also passed:
**49 controller tests + 8 independent acceptance-logic tests**. No dependencies
were installed and no GPU/physics rollout was launched for publication. The
published source/protocol/configuration matched all **54** relevant recorded
runtime input hashes. All four media digests matched their original receipt;
both MP4s fully decoded to 971 frames and the GIF to 194 frames / 19.4 seconds.
The five updated public documentation pages passed 69 relative-link checks.
These are local publication checks, not a claim of hosted CI success.

The current policy is a Transformer actor-critic trained with PPO, not a CVAE.
The latest v3 retargeting experiment completed 16/16 jobs, with mean wrist-target
error reduced to 0.540 degrees and no output hinge-rate violations in the combined
arm. It is four-clip IK evidence, not improved learned behavior. The original
7,174-clip training index and official default policy were not replaced. Autonomous
navigation, learned box carrying and a full BFM reproduction are not complete.
See [the result and caveats](UMR_STAGE2_CONTROLS_V3_20260909.md).

No root LICENSE is invented by this publication. Existing source notices remain
in place, including `ScaleRetarget/scaleretarget/utils/lafan_vendor/license.txt`.
Upstream root licensing and broader redistribution review remain unresolved;
package metadata alone does not resolve them. See
[third-party notices](../THIRD_PARTY_NOTICES.md) and
[the release checklist](RELEASE_CHECKLIST.md).
