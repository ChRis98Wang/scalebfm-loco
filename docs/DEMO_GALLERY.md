# Demonstration gallery

The README contains **13 complete recordings**, with **13 GIFs, 26 MP4s and
13 posters** (78,158,772 bytes of media). The current native clips appear first;
earlier comparisons and failed tasks are labelled separately. This is a gallery
of recorded experiments, not 13 independently learned behaviors or a task-success
rate.

All clips use the **official pretrained ScaleBFM policy**. The native checkpoint
is `model_22200.pt` (SHA-256
`88d5a79946c03ed25503f48b2af71d16290844ef066ca9b6c8fa8dc3837422e3`);
MuJoCo clips use its previously verified `export_v2` deployment. No policy was
trained or promoted for this publication.

## All playable files

| Recording | Engine | Result / scope | Length | GIF | Captioned MP4 | Raw MP4 |
| --- | --- | --- | ---: | --- | --- | --- |
| Box lift, hold and replace | IsaacLab / PhysX | single state pass | 19.42 s | [GIF](media/isaac-lift-single-state-20260910.gif) | [MP4](media/isaac-lift-single-state-20260910.mp4) | [Raw](media/isaac-lift-single-state-20260910-raw.mp4) |
| Squat and stand | IsaacLab / PhysX | preview pass | 4.58 s | [GIF](media/isaac-motion-squat-20260910.gif) | [MP4](media/isaac-motion-squat-20260910.mp4) | [Raw](media/isaac-motion-squat-20260910-raw.mp4) |
| Cha-cha | IsaacLab / PhysX | preview pass | 8.24 s | [GIF](media/isaac-motion-chacha-20260910.gif) | [MP4](media/isaac-motion-chacha-20260910.mp4) | [Raw](media/isaac-motion-chacha-20260910-raw.mp4) |
| Waltz | IsaacLab / PhysX | preview pass | 7.06 s | [GIF](media/isaac-motion-waltz-20260910.gif) | [MP4](media/isaac-motion-waltz-20260910.mp4) | [Raw](media/isaac-motion-waltz-20260910-raw.mp4) |
| Walk, turn and return | IsaacLab / PhysX | preview pass | 10.58 s | [GIF](media/isaac-motion-turn-20260910.gif) | [MP4](media/isaac-motion-turn-20260910.mp4) | [Raw](media/isaac-motion-turn-20260910-raw.mp4) |
| Walk and reach low (empty-handed) | IsaacLab / PhysX | preview pass | 6.90 s | [GIF](media/isaac-motion-reach-low-20260910.gif) | [MP4](media/isaac-motion-reach-low-20260910.mp4) | [Raw](media/isaac-motion-reach-low-20260910-raw.mp4) |
| Standing 3D wrist targets | MuJoCo | engineering pass | 12.00 s | [GIF](media/synthetic-wrist-control-vr3-20260909.gif) | [MP4](media/synthetic-wrist-control-vr3-20260909.mp4) | [Raw](media/synthetic-wrist-control-vr3-20260909-raw.mp4) |
| Movement with changing wrist height | MuJoCo | engineering pass | 20.00 s | [GIF](media/synthetic-move-wrists-20260909.gif) | [MP4](media/synthetic-move-wrists-20260909.mp4) | [Raw](media/synthetic-move-wrists-20260909-raw.mp4) |
| Earlier squat and stand | MuJoCo | preview pass | 4.60 s | [GIF](media/motion-squat-20260909.gif) | [MP4](media/motion-squat-20260909.mp4) | [Raw](media/motion-squat-20260909-raw.mp4) |
| Earlier cha-cha | MuJoCo | preview pass | 8.26 s | [GIF](media/motion-chacha-20260909.gif) | [MP4](media/motion-chacha-20260909.mp4) | [Raw](media/motion-chacha-20260909-raw.mp4) |
| Earlier waltz | MuJoCo | preview pass | 7.08 s | [GIF](media/motion-waltz-20260909.gif) | [MP4](media/motion-waltz-20260909.mp4) | [Raw](media/motion-waltz-20260909-raw.mp4) |
| Box push — strict task FAIL | MuJoCo | failed trial | 20.00 s | [GIF](media/loco-push-20260909.gif) | [MP4](media/loco-push-20260909.mp4) | [Raw](media/loco-push-20260909-raw.mp4) |
| Earlier box lift — FAIL | IsaacLab / PhysX | failed trial | 11.84 s | [GIF](media/isaac-lift-balanced-failed-20260910.gif) | [MP4](media/isaac-lift-balanced-failed-20260910.mp4) | [Raw](media/isaac-lift-balanced-failed-20260910-raw.mp4) |

## What the clips demonstrate

- **Box lift, hold and replace:** 1 kg; 10.19 cm clearance; 2.02 s hold; 1.78 cm final XYZ error. One initial state, not walking carry.
- **Squat and stand:** 2.73 cm mean body-position error; empty-handed.
- **Cha-cha:** 5.12 cm mean body-position error; selected development clip.
- **Waltz:** 7.61 cm mean body-position error; selected development clip.
- **Walk, turn and return:** 6.41 cm mean body-position error; turning route, not arbitrary-goal navigation.
- **Walk and reach low (empty-handed):** 6.74 cm mean body-position error; no box is present despite the reference filename.
- **Standing 3D wrist targets:** VR-3 wrist XYZ RMSE 3.54 / 3.77 cm; single initial state, no object contact.
- **Movement with changing wrist height:** 0.9295 m forward travel; only 1 of 4 paired development cases passed. No object carrying.
- **Earlier squat and stand:** Earlier simulation comparison, not an additional motion type or new policy.
- **Earlier cha-cha:** Earlier simulation comparison, not an additional motion type or new policy.
- **Earlier waltz:** Earlier simulation comparison, not an additional motion type or new policy.
- **Box push — strict task FAIL:** 17/19 checks passed; right-wrist contact continuity and floor support failed. Not carrying.
- **Earlier box lift — FAIL:** Stopped on excessive box tilt; did not lift clear of the table. Earlier failed trial.

The five native body-motion clips passed their fixed numerical preview gates;
they were selected development references, not random held-out samples. Body
tracking, empty-handed reaching and wrist-target following do not establish
object manipulation. The box lift is a **single-initial-state** physical pass;
walking carry, formal multi-initial-state acceptance and hardware calibration
remain incomplete. [Box-lift evidence and controller explanation](ISAAC_BOX_LIFT_DEMO_20260910.md).

The policy does not consume camera images. The box-lift planner obtains
simulator object/contact state and supplies sparse body targets; the learned
controller supplies joint actions. The other synthetic previews use generated
body targets. Videos are policy/PD physics rollouts, not frame-by-frame robot or
object pose animation.

## Media integrity and recording boundaries

- Every original and captioned MP4 was fully decoded and matched the expected
  recorded frame count. All 13 GIFs were fully decoded and loop at original speed.
- The 3 newly generated GIFs (standing wrists, moving wrists and pushing) are
  complete downsampled versions of the existing captioned videos, with captions
  retained. Existing MP4s were not re-encoded or cut.
- GIFs are 640x480 / 10 fps, with at most one display-frame duration difference.
  MP4s are 50 fps; native/body/push clips are 960x720 and synthetic wrist previews
  are 640x480.
- Native clips record every physical control step, starting after reset-only
  initialization. Earlier MuJoCo clips additionally retain their initialization
  frame. This explains the 0.02 s video-duration difference for matching motions.
- Raw media hashes match the original recording reports. Existing archive
  receipts retain their historical local-only/publication flags unchanged.
- Previous visual QA sampled frames; **no complete manual frame-by-frame review
  or new physical rollout is claimed by this media publication**.

[Machine-readable catalog, hashes and decode counts](media/gallery-20260910.json)
contains all 52 media files. The light-weight repository integrity tests run with:

```bash
python -B -m unittest discover -s tests -p test_demo_gallery.py
```

Incomplete or stale-renderer retries, duplicate source copies and raw FK/retargeter
debug videos are not presented as usable policy demos. Source datasets, model
weights, robot meshes and complete experiment logs remain excluded.

## Sources and further development

Rendered media are published as a non-commercial research presentation at the
repository owner's request. The human-motion-derived clips retain AMASS, KIT and
ACCAD provenance; neither those datasets nor their motion arrays are published.
See [media credits and source terms](MEDIA_SOURCES.md). This is not a new blanket
license or a claim that all underlying rights have been cleared.

[AMASS workflow](AMASS_TO_SCALEBFM_TECHNICAL_ROUTE.md) ·
[GUI / motion switching](AMASS_MOTION_BROWSER.md) ·
[Loco-manipulation acceptance plan](LOCO_MANIPULATION_DELIVERY_PLAN.md)
