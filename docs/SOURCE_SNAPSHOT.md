# Source snapshot — 2026-09-09

This is a development source snapshot, not a complete ScaleBFM reproduction,
rights-cleared release, or standalone simulator installation.

## Provenance and contents

- Destination: `ChRis98Wang/scalebfm-loco`, branch `publish/scalebfm-20260909`.
- Parent: the destination's initial README commit
  `029659cebeb751a55f812153df8b0b9ac8eaf343`; existing `main` is preserved.
- Upstream source baseline: [ScaleBFM at abd6f17](https://github.com/zengweishuai/ScaleBFM/tree/abd6f17c02fe0baabc14709feb8d9ea4959aa621).
- Local engineering base: `897b1460146def06c705253dfee7c48db509c530`, plus the
  audited working-tree changes through the Stage II v3 experiment.
- Included: main Python modules, shell entry points, configuration, packaging
  metadata, tests, CI and engineering reports, with existing author/license headers.
- Excluded: the local asset-bearing Git history, external UMR checkout, Unitree
  SDK, robot models/meshes, bundled wheels, examples/motion arrays, AMASS/SMPL-X
  inputs, checkpoints, logs, caches, credentials and media.

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
git clone --branch publish/scalebfm-20260909 https://github.com/ChRis98Wang/scalebfm-loco.git /path/to/scalebfm-loco
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
