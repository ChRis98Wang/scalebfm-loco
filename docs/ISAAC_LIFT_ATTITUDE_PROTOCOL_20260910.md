# Stronger pitch feedback with unchanged physical gates

Fixed before the next trial, 2026-09-10. Trial h tested the previously prepared
clearance feedback and failed at 12.04 s: box pitch exceeded 15 degrees before
meaningful clearance was established. Increasing height alone did not solve
the rear-biased grasp's tipping tendency.

Profile `isaac_lift_balance_attitude_20260910.json` retains the forward approach
and 1/s clearance loop, and changes wrist-pitch feedback gain 1.5→3.0 with a
correction limit 0.25→0.35 rad (within the existing declared config workspace).
No change to mass, geometry, friction, solver, PD, torque limits, force gate,
8 cm / 2 s / return-and-release criteria, or 20 s task timeout. Only sparse
reference orientations are requested; actual robot/box poses remain measured.

This is a development hypothesis, not proof of stable lift. First run unrecorded
for diagnosis, in a finite owned unit with policy/inputs frozen and failures kept.
