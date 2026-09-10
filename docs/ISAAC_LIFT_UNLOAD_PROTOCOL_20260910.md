# Faster contact-gated unloading, unchanged acceptance

2026-09-10, fixed before execution. Trial j retained successful airborne hold
and avoided the 40 N peak failure, but still failed LOWER's original 4 s timeout:
both arms retained about 8 N conservative contact force and the table did not
carry the required stable conservative force/velocity window. Recentring XY was
executed, but no completed place/release is claimed.

Keep the same landing trigger and all task/scene/physics/policy settings. After
measured landing contact only, request 1 N instead of 6 N and use contact-force
feedback gain 0.012 instead of 0.004 m/(N*s). The existing **6 cm/s maximum gap
rate**, gap workspace, bilateral loss guard, 40 N peak cap and stable support
0.8 mg / 0.2 s before release are unchanged. Airborne force target stays 24 N.

This changes the rate of unloading within the same target-motion speed bound;
it does not reduce physical acceptance thresholds. Loss of grip before stable
support, unstable placement, force limits or any timeout still fails the trial.
