# ScaleBFM-Loco — engineering source snapshot

## Native IsaacLab box lift — single-state development pass

[![Official BFM plus VR-3 planner lifts a physical 1 kg box, holds, replaces and releases it; single initial state, no vision](docs/media/isaac-lift-single-state-20260910.gif)](docs/ISAAC_BOX_LIFT_DEMO_20260910.md)

[Open GIF](docs/media/isaac-lift-single-state-20260910.gif) ·
[Captioned MP4](docs/media/isaac-lift-single-state-20260910.mp4) ·
[Original MP4](docs/media/isaac-lift-single-state-20260910-raw.mp4) ·
[Technical explanation, evidence and limits](docs/ISAAC_BOX_LIFT_DEMO_20260910.md)

The **official pretrained BFM + VR-3 task planner** lifts a dynamic **1 kg** box
through real contact in **IsaacLab / PhysX**: maximum measured bottom clearance
**10.19 cm**, a **2.02 s hold**, **1.78 cm** final XYZ error, and stable release
within **19.42 s**. The GIF plays inline at original speed; all 971 physical
control steps are retained in the MP4. No fingers, attachment, object animation
or extra lifting force are used.

**No vision input:** the planner reads simulator ground-truth box pose and
contact forces through IsaacLab APIs and supplies pelvis / left-wrist /
right-wrist XYZ and orientation targets. The Transformer policy consumes robot
proprioception, target/mask information and action history, not camera images or
box state directly. It outputs joint actions; PD actuators and PhysX produce the
motion. The renderer is used only to record the demonstration.

Ten independent physical sequence checks passed locally. The recorded and
unrecorded runs have identical trajectories, but use **the same initial state
and seed**. This is not formal 60-episode D1 acceptance, walking carry, autonomous
visual manipulation, hardware validation, or a newly trained manipulation policy.
Materials are not hardware-calibrated; this experiment performed **zero training
updates**. The [controller](ScaleTrack/scripts/pretrain/rsl_rl/lift_demo_balanced.py),
[native runner](scripts/run_bfm_isaac_lift_balanced.py),
[independent checks](scripts/bfm_lift_acceptance.py) and 57 logic tests are included.

## Source and publication scope

The `main` branch publishes the ScaleRetarget / ScaleTrack / ScaleBridge
source, local extensions, tests and engineering reports. It continues the initial
README-only project page; it is **not a complete, asset-bundled release**.
Robot assets, bundled SDKs, motion data, body models, weights and experiment logs
are deliberately excluded. The explicit media exception is the robot-FK-only
box-lift demo above: GIF, captioned/raw MP4, poster and archive receipt. See
[snapshot scope and runtime setup](docs/SOURCE_SNAPSHOT.md) before using the
upstream setup instructions below. Reuse an existing IsaacLab environment.

High-dynamic controller recordings remain pending motion-source publication
review; KIT/ACCAD-derived dance and body-motion videos are not included in this
publication. Local report paths and experiment hashes describe the original
development machine, not downloadable evaluation artifacts.

This is an **unofficial reproduction and engineering extension** of
[zengweishuai/ScaleBFM](https://github.com/zengweishuai/ScaleBFM), the upstream
implementation of *Scaling Behavior Foundation Model for Humanoid Robots*.
Upstream authorship and third-party notices are retained. The description below
summarizes the upstream project, not capabilities fully reproduced by this fork.

> ScaleBFM investigates how Behavior Foundation Models can be effectively scaled
> through the coordinated design of the learning paradigm, behavioral data, and
> model architecture. The resulting framework enables humanoid robots to perform
> diverse behaviors with natural whole-body coordination, including agile
> locomotion, dexterous manipulation, and coordinated loco-manipulation in both
> simulation and the real world.

<p align="center">
  <a href="https://scalebfm.github.io/"><img src="https://img.shields.io/badge/Project-Website-4285F4?logo=googlechrome&logoColor=white" alt="Project website"></a>
  <a href="https://arxiv.org/abs/2607.15163"><img src="https://img.shields.io/badge/arXiv-2607.15163-b31b1b?logo=arxiv&logoColor=white" alt="arXiv paper"></a>
  <a href="https://github.com/zengweishuai/ScaleBFM/issues"><img src="https://img.shields.io/badge/GitHub-Issues-181717?logo=github&logoColor=white" alt="GitHub issues"></a>
</p>

## Local AMASS end-to-end workflow

This workspace includes a safe, resumable AMASS-to-ScaleBFM entry point with
separate ScaleRetarget and IsaacLab interpreters, opt-in training/playback, and
runtime quaternion compatibility. See the Chinese
**[AMASS → ScaleBFM technical route and implementation guide](docs/AMASS_TO_SCALEBFM_TECHNICAL_ROUTE.md)**.

The current policy is a **Transformer actor-critic trained with PPO**, not a CVAE.
Local work uses official pretrained weights plus AMASS fine-tuning; it is not
from-scratch BFM pretraining or a claim of full paper reproduction. CMU is not a
prerequisite for the existing ACCAD/BMLmovi/BMLrub/CNRS workflow.

[KIT ingestion](docs/KIT_INGESTION_20260907.md) is now complete: 4,231 validated
robot clips, with 2,927 added to a new 7,174-clip training index and 789 kept in a
separate KIT holdout. The old 962-clip benchmark remains unchanged. A bounded
one-update smoke used 55 KIT clips in actual rollouts; this is integration
evidence, not successful training of every clip or a promoted checkpoint.
New [PPO diagnostics](docs/SCALEBFM_PPO_DIAGNOSTICS_20260907.md) expose within-update
KL, ratio clipping, and actual learning-rate ranges without changing the optimizer.

Current development has returned to **behavior learning**, with GUI/hand-written
waypoint extensions paused. The [learning-mainline protocol](docs/BEHAVIOR_LEARNING_MAINLINE_20260908.md)
records a completed five-update fixed-vs-adaptive LR pilot (the full eight-mask
evaluation queue was not completed), then a user-requested 1,000-update coverage
experiment. The opt-in sampler rotates 128 clips every five PPO updates through
the 7,174-clip training pool; the 962 legacy and 789 KIT development-validation
clips stay out of gradient updates. Loading or sampling the full pool does not
mean every clip has been sufficiently learned; no candidate is automatically
promoted, and the protocol page records actual execution status. The 1,000-update
run has now finished with all 7,174 training clips actually sampled. Its
[independent eight-mask evaluation](docs/BEHAVIOR_LONG_EVALUATION_20260908.md)
has also completed: the candidate passed 0/8 aggregate and 0/16 stratum-by-mask
no-regression checks against freshly rerun official weights. Mean active-link
position error increased by 1.07–6.15% across modes. The official default is
unchanged; successful training and complete sampling did not establish improved
tracking quality.

The next mainline is [versioned data-quality improvement and completion gates](docs/DATA_REFRESH_AND_SCALEBFM_COMPLETION.md):
an optional pinned **unofficial UMR** backend, an experimental true SMPL-X surface
adapter, and a 40-clip origin-preserving pilot. Collision-sole height correction
is opt-in; new references are not automatically accepted for training. Existing
data and official default weights are preserved until physical tracking and
paired training evaluations support a replacement.
The completed 40-clip frozen-policy A/B rejected the height-only candidate:
0/8 aggregate no-regression gates passed. The separate AMASS/SMPL-X-to-UMR-to-IsaacLab
single-clip smoke runs end to end; it is not evidence of better training data yet.
The expanded [true-UMR same-origin evaluation](docs/UMR_PAIRED_EVALUATION_20260909.md)
is now complete: 40 paired clips, 80 validated archives and all 16 policy runs.
Foot penetration improved, but the combined no-regression gate passed 0/8 modes;
WholeBody-14 mean position error increased from 5.048 to 5.711 cm. This does not
establish better training data; that frozen-policy batch performed no PPO updates
and its owned jobs exited. The [resume handoff](docs/UMR_RESUME_HANDOFF_20260908.md)
records preserved interrupted artifacts and the explicit paired-clock packaging fix.

The [deployment/Sim2Sim audit](docs/SCALEBFM_SIM_CHAIN_20260909.md) now connects the
official checkpoint to a verified, TensorRT-free TorchScript policy and bounded
native ScaleBridge/MuJoCo rollouts. All 16 numerical export checks passed. The
height-preserving local-control variant completed 48 runs (three short references,
eight masks, global/local), with 47/48 passing the predeclared tracking guard and
no fall-guard or numerical-warning events. The full quality gate is still false;
this is not a newly trained policy, arbitrary-goal navigation or a manipulation
success claim. Original assets/default weights remain unchanged. Data-adaptation
and task-delivery boundaries are defined in the [paired UMR learning protocol](docs/UMR_BEHAVIOR_AB_PROTOCOL_20260909.md) and
the staged [loco-manipulation demo plan](docs/LOCO_MANIPULATION_DELIVERY_PLAN.md).

The separate [UMR learning execution record](docs/UMR_BEHAVIOR_AB_RUN_20260909.md)
now records a new 256-origin matched A/B dataset (17 paired targets + 239 identical
old-reference replay clips). Both six-update engineering smokes passed with matched
actual cohorts and verified Adam updates. Formal 100-update-per-arm training also
completed from the common initial checkpoint: 1,638,400 total environment steps,
with 3,200 actual steps per origin in each arm. All 56 evaluation jobs completed,
but the original aggregate no-regression gate passed only 2/8 modes for B vs A
and 0/8 for B vs official (strata: 2/16 and 0/16). This version is not promoted,
and the predeclared seed-43/44 confirmation condition was not met. All owned jobs
exited. The
original 7,174-motion index and official default weights are not replaced.

A subsequent [wrist-target audit and canonical-hand v2 experiment](docs/UMR_WRIST_AND_HAND_V2_20260909.md)
completed all 27 diagnostic origins and a fixed four-origin, eight-job retargeting
comparison. Matching the canonical hand to the constant source hand reduced mean
historical wrist-target discrepancy from 45.735° to 29.192°, but one jogging clip
regressed and another clip gained 4.27 mm self-penetration. This is a mechanism
experiment, not a policy improvement or data promotion; no PPO was run in this
stage. All jobs exited, and the existing data/default model remain unchanged.

The next [Stage II wrist-orientation/output-rate v3 experiment](docs/UMR_STAGE2_CONTROLS_V3_20260909.md)
completed all 16 fixed-sample jobs. Adding both controls reduced mean wrist-target
error to 0.540° (control: 45.735°) and kept all output hinge rates at or below
12 rad/s. Surface-normal error increased and small self-overlap remains: this
is four-clip IK/FK evidence, not policy tracking, new behavior learning or a full
data promotion. All 1,185 local regression tests passed; owned jobs exited.

The [2026-09-07 paired evaluation](docs/BFM_EVALUATION_20260907.md) found a
20.10% increase in mean WholeBody-14 position error after local fine-tuning on
the held-out local clips. The final checkpoint is not an established improvement
over the official baseline. A subsequent
[eight-mask, 512-environment paired evaluation](docs/SCALEBFM_MASK_RESULTS_20260907.md)
also found regression in all eight modes. Active-link metrics and a strict report
comparer are now implemented; this is single-seed offline-reference evidence,
not validation of arbitrary online goals or object tasks. See the
[mask validation protocol](docs/SCALEBFM_MASK_VALIDATION.md).

### Native IsaacLab GUI

Reuse an existing IsaacLab environment. From the repository root:

```bash
export BFM_ISAAC_PYTHON=/path/to/your/existing/isaaclab/bin/python
bash scripts/open_motion_browser.sh
```

The launcher opens the actual IsaacLab/Kit robot simulation and motion menu.
It requires local checkpoints and motion indexes, which are not bundled in a
source release. On the development machine the existing interpreter is detected
at its conventional workspace location; on other machines set the variable above.
See [motion browser usage](docs/AMASS_MOTION_BROWSER.md) for overrides and cleanup.

For online pelvis/wrist targets in the native Kit panel:

```bash
BFM_LOAD_RUN=humanoid_transformer_m BFM_CHECKPOINT=model_22200.pt \
BFM_INITIAL_MOTION=ACCAD/Male2General_c3d/A1-_Stand_stageii \
bash scripts/open_motion_browser.sh --online-targets
```

The panel supports **Pelvis-1 / UMI-2 / VR-3** with the same Transformer policy.
Switching modes pauses physics; resume explicitly and submit a fresh Apply.
Mode changes update reference masks, not robot state. This remains a constrained
online-control interface, not autonomous navigation or box carrying. See
[online operation and safety limits](docs/ONLINE_VR3_CONTROL.md).

The Pelvis-1 panel also exposes an experimental **XYZ + yaw waypoint** (Z is
pelvis height), with bounded future references and actual-state arrival/stall
checks. Real actor-input diagnostics now confirm correct XYZ/mask delivery.
The height-only diagnostic passed the existing 3-cm height tolerance (2.68-cm
residual), including a 2-second live hold. Forward-only and combined targets
still **fail by stalling**; this is not validated waypoint locomotion. See the
[controls](docs/ONLINE_WAYPOINT_CONTROL.md) and
[real-physics diagnostics and results](docs/ONLINE_WAYPOINT_DIAGNOSTICS_20260908.md).

A subsequent [native KIT gait check](docs/KIT_GAIT_VALIDATION_20260908.md) passed
two walking clips in both Pelvis-1 and WholeBody-14 with the same official policy:
2.27–2.42 m of actual forward travel and 2.62–4.48 cm mean pelvis error, with no
in-clip resets or direct state writes. These are short-clip pelvis/foot kinematic
checks, not all-body error or contact-stability validation. Native references
exceed the online speed/lead limits; adapting them to XYZ waypoint control is
still pending. The linked report includes a local two-clip GUI preview command.

For a physical target box and live reference-mode switching:

```bash
bash scripts/open_motion_browser.sh --target-object
```

The optional scene adds gravity, collision, configurable contact parameters,
target-only reset, and actual object telemetry. Pelvis-1 / VR-3 / WholeBody-14
can be switched while browsing motions. See the
[target-object validation guide](docs/BFM_TARGET_OBJECT_VALIDATION.md).
The contact parameters are engineering defaults, not real-material calibration;
the current checkpoint is not an object-conditioned grasping or navigation policy.
The [capability matrix and implementation order](docs/BFM_CAPABILITY_MATRIX.md)
distinguishes upstream source availability, local runtime checks, and task-level
validation. This fork does not yet reproduce the complete BFM capability set.

### Development and release status

See [contributing and testing](CONTRIBUTING.md) and the
[release checklist](docs/RELEASE_CHECKLIST.md) and
[third-party notices](THIRD_PARTY_NOTICES.md). This checkout is not yet a
license-cleared public release. No dataset/model redistribution rights are
implied by this README, package metadata, or ignore rules.
Simulator-independent GitHub Actions tests are configured separately from
licensed-data and GPU/Kit integration checks; no hosted run is claimed yet.

## ScaleRetarget

**ScaleRetarget** is the motion-retargeting toolkit used to prepare human motion
data for ScaleBFM. It converts motion from common mocap representations into
robot joint trajectories and currently provides a complete configuration for the
Unitree G1 humanoid.


The retargeting code and documentation are available in
**[ScaleRetarget](ScaleRetarget/README.md)**. See the guide for supported datasets,
environment setup, motion retargeting, dataset preparation, and visualization.

Retargeting media are available in the
[upstream demonstrations](https://github.com/zengweishuai/ScaleBFM/tree/abd6f17c02fe0baabc14709feb8d9ea4959aa621/ScaleRetarget#readme),
not as recordings produced by this fork.

## ScaleTrack

**ScaleTrack** provides the foundational implementation of the BFM pretraining
pipeline. It packages retargeted robot trajectories into motion datasets and
pretrains BFMs through motion tracking in IsaacLab using the bundled RSL-RL
implementation. It currently provides a complete pretraining pipeline for the
Unitree G1 humanoid, including single- and multi-GPU training, policy playback, and export.

The training code and documentation are available in
**[ScaleTrack](ScaleTrack/README.md)**. See the guide for environment and robot
asset setup, motion preparation and packaging, policy training, playback, and
export.

Tracking media remain in the
[upstream demonstrations](https://github.com/zengweishuai/ScaleBFM/tree/abd6f17c02fe0baabc14709feb8d9ea4959aa621/ScaleTrack#readme).

## ScaleBridge

**ScaleBridge** provides a unified Sim2Sim and Sim2Real deployment framework for
Behavior Foundation Models on humanoid robots. It enables a seamless transition
from policy evaluation in MuJoCo to deployment on physical robots.

The deployment code and documentation are available in
**[ScaleBridge](ScaleBridge/README.md)**. See the guide for environment setup,
robot controller configuration, policy evaluation, real-world deployment, and
migration instructions.

Deployment media remain in the
[upstream demonstrations](https://github.com/zengweishuai/ScaleBFM/tree/abd6f17c02fe0baabc14709feb8d9ea4959aa621/ScaleBridge#readme).

## Citation

If you find the upstream work useful, please cite its paper and previous work
that introduced the BFM framework:

```bibtex
@article{zeng2026scaling,
  title   = {Scaling Behavior Foundation Model for Humanoid Robots},
  author  = {Zeng, Weishuai and Yin, Kangning and Niu, Xiaojie and
             Lu, Shunlin and Zhong, Weixiang and Chen, Jiahe and
             Jia, Feiyu and Chen, Xiao and Wang, Zirui and Xu, Furui and
             Zhou, Ming and Li, Kailin and Zhang, Weinan and Wang, He and
             Yi, Li and Lin, Dahua and Pang, Jiangmiao and Wang, Jingbo},
  journal = {arXiv preprint arXiv:2607.15163},
  year    = {2026}
}
```

```bibtex
@article{zeng2025behavior,
  title   = {Behavior Foundation Model for Humanoid Robots},
  author  = {Zeng, Weishuai and Lu, Shunlin and Yin, Kangning and
             Niu, Xiaojie and Dai, Minyue and Wang, Jingbo and Pang, Jiangmiao},
  journal = {arXiv preprint arXiv:2509.13780},
  year    = {2025}
}
```
