# Third-party source notices

This is a partial inventory, not a root license or a completed redistribution
audit. No repository-wide MIT grant is made by this document. Package metadata
does not override file-level notices or third-party terms.

## ScaleBFM

The main ScaleRetarget, ScaleTrack and ScaleBridge layout and implementation
originate from [zengweishuai/ScaleBFM](https://github.com/zengweishuai/ScaleBFM),
baseline `abd6f17c02fe0baabc14709feb8d9ea4959aa621`. Upstream authorship is retained.
The baseline has no root LICENSE; resolving the scope of permission remains a
[release prerequisite](docs/RELEASE_CHECKLIST.md), not something this fork can
resolve by assigning its own package-wide license.

## ETH Zurich / NVIDIA source

Files in `ScaleTrack/source/my_rsl_rl/` that identify BSD-3-Clause retain their
original `Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION` headers.
The accompanying [BSD-3-Clause terms](licenses/BSD-3-Clause-ETH-NVIDIA.txt) apply
to those identified portions, not automatically to every file in that directory.
Existing source notices take precedence if their copyright years differ.

`ScaleRetarget/scaleretarget/utils/kinematic_model/torch_utils.py` retains the full
NVIDIA 2018-2022 BSD-style copyright, conditions and disclaimer in its source.

## LAFAN vendor

`ScaleRetarget/scaleretarget/utils/lafan_vendor/` is copied unchanged from the
fixed upstream baseline, including its complete
[CC BY-NC-ND 4.0 license text](ScaleRetarget/scaleretarget/utils/lafan_vendor/license.txt).
It is not relicensed as MIT and is not motion data. The source is available in
the [original vendor directory](https://github.com/zengweishuai/ScaleBFM/tree/abd6f17c02fe0baabc14709feb8d9ea4959aa621/ScaleRetarget/scaleretarget/utils/lafan_vendor).
Review those terms independently before using or redistributing this component.

## Not bundled

Unitree SDK/binaries, robot assets, AMASS/SMPL-X data and models, external UMR,
and checkpoints are not included. The only rendered-media exception is the
[synthetic-reference box-lift demonstration](docs/ISAAC_BOX_LIFT_DEMO_20260910.md),
which uses the official pretrained policy and robot-only FK targets, not human
motion frames. Publishing these rendered results does not grant redistribution
rights to the underlying geometry, policy weights, simulator or data.
Obtain any required dependency directly
from its original source and preserve its own terms and notices. Exclusion from
this source tree is not a grant of rights to publish those dependencies later.
