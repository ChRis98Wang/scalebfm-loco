# Forward-centred grip development configuration

Fixed before the next run, 2026-09-10. Reuses the v4 balanced runner and all
physical/acceptance settings; only `wrist_offset_x_m` changes −0.03 → +0.03 m.
The existing pre-contact X bias cap, rate limits, frozen XY anchor, attitude
feedback and original 18 N/0.20 s preload gate are unchanged.

In balanced trial d, the measured final pair-average contact positions were
6.48 cm (left) / 5.92 cm (right) behind box centre, and box pitch still exceeded
15°. The hypothesis is that an additional 6 cm nominal wrist-X request moves
load-bearing contacts toward the side centres. This is a controller request,
not a measured contact shift or success claim. No box mass/friction/PD changes.

First run is unrecorded for quick diagnosis, still bounded to 20 physical seconds
and the full task sequence, with all per-step contact data and failed runs saved.
If useful, repeat with full original-speed recording and compare trajectories.
