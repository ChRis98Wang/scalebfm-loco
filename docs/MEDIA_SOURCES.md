# Media sources and publication boundaries

The gallery contains this project's **rendered robot-simulation research results**,
not camera recordings of mocap participants or downloadable motion datasets.
The owner requested publication of the complete playable demo gallery on
2026-09-10. Existing source/asset terms are not replaced by that request, the Git
repository, or its package metadata.

## Policy and robot

Credit for the pretrained behavior model belongs to the
[ScaleBFM authors and upstream project](https://github.com/zengweishuai/ScaleBFM).
The robot is Unitree G1, using the upstream assets in the local simulation setup.
IsaacLab / PhysX and MuJoCo results are labelled separately. No policy checkpoint,
robot mesh, body model, SDK binary or full simulation log is distributed here.

## Human-motion references

| Published recording families | Reference source |
| --- | --- |
| Native and earlier squat/stand | KIT `3/squat01`, via AMASS and local robot retargeting |
| Native and earlier cha-cha | KIT `572/dance_chacha01`, via AMASS and local robot retargeting |
| Native and earlier waltz | KIT `572/dance_waltz00`, via AMASS and local robot retargeting |
| Native walk/turn/return | ACCAD Female1Walking B15, via AMASS and local robot retargeting |
| Native empty-handed low reaching | ACCAD Female1Walking B19, via AMASS and local robot retargeting |
| Box lift, earlier failed lift, push, standing/moving wrist targets | Robot-only FK initialization and synthetic targets; no human motion input for these trials |

The last row describes the demonstration references, not the training-data
provenance of the upstream pretrained model.

- **AMASS:** Mahmood et al., ICCV 2019, *AMASS: Archive of Motion Capture as Surface
  Shapes*. Its source license permits specified non-commercial research,
  education and artistic uses and restricts distribution of the dataset.
  We do not publish AMASS arrays, fitted body parameters or source software as
  part of this media update. [Official AMASS license and citation](https://amass.is.tue.mpg.de/license.html).
- **KIT:** credit the KIT Whole-Body Human Motion Database and Mandery et al.
  (2016), *Unifying Representations and Large-Scale Whole-Body Motion Databases for
  Studying Human Motion*. The database's publication guidance identifies the
  appropriate references; it is not a blanket license for this repository.
  [Official KIT database and citation guidance](https://motion-database.humanoids.kit.edu/).
- **ACCAD:** credit the Open Motion Project, Advanced Computing Center for the
  Arts and Design, The Ohio State University. The source project identifies
  CC BY 3.0 for its published mocap collection. Our changes are AMASS-based
  processing, robot retargeting, policy-driven simulation and video captions.
  This source attribution does not override the AMASS processing-stage terms.
  [Official ACCAD source and credit](https://accad.osu.edu/research/motion-lab/mocap-system-and-data).

## What publication does not grant

This is a non-commercial research presentation, **not** a commercial-use
clearance, a redistribution of the source datasets, or an automatically granted
license to every source asset/model or downstream use of the footage.
The linked source terms and file-level notices remain authoritative; review
the relevant permissions separately before commercial use or republishing.

The original local archive JSONs keep `publication_performed=false` and
`redistribution_rights_verified=false` where recorded. They document the earlier
archival step and are not retroactively changed into a legal-clearance receipt.
The later publication is recorded by Git history and
[the current gallery catalog](media/gallery-20260910.json).

[Gallery and limitations](DEMO_GALLERY.md) ·
[Third-party source notices](https://github.com/ChRis98Wang/scalebfm-loco/blob/main/THIRD_PARTY_NOTICES.md)
