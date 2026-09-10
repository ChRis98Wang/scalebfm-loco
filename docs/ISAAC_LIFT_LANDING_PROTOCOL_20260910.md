# Contact-gated unloading and return to the declared XY goal

2026-09-10, fixed before execution. Trial i reached 10.19 cm minimum-corner
clearance, passed the 2 s HOLD, and failed during LOWER at 17.34 s on a measured
40.027 N substep left clamp peak. The box had landed but the 24 N clamp controller
was still tightening against intermittent minimum contact readings. Final box Y
was 3.61 cm from the declared target, also outside the final 3 cm XYZ tolerance.

The landing profile preserves trial i unchanged through the airborne hold.
During LOWER only, after conservative table force ≥0.5 mg, lowest corner within
5 mm of table, tilt ≤10 degrees and box speed ≤0.10 m/s, latch unloading and
reduce only the clamp **control setpoint** to 6 N. Existing 6 cm/s side-gap slew,
workspace bounds, 40 N peak cap, bilateral contact and 0.8 mg stable support gates
remain. A target-force change is not a claim of measured support/release success.

During the original 2 s lower reference, smoothly move the frozen pickup XY
anchor to the declared table goal XY. RELEASE opens around that same goal. The
box is moved only by robot contact; no object pose, force, joint or trajectory
is assigned by the planner. Original lower Z reference and contact constraints
are unchanged, as are the scene, 1 kg mass, material, solver, PD and policy.

Run full 20 s maximum sequence. Keep the original 8 cm / 2 s / 1 s stable-release
and all other physical failure gates. First run unrecorded; failures retained.
No formal D1/D2 or generalisation claim follows from one development initial state.
