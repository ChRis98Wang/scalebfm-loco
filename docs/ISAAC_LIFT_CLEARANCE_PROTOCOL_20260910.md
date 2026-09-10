# Measured-clearance feedback after centred approach

2026-09-10, fixed before execution. Trial f (forward wrist X) measured actual
minimum-corner clearance 1.8847 cm and zero support force at the end; tilt was
13.98 degrees. It still failed the unchanged 8 cm minimum within 3 s of LIFT.

New profile `configs/isaac_lift_balance_clearance_20260910.json` retains the
forward approach and all scene, physics, force and task gates. It adds an outer
loop **only after the original load-bearing preload is established**: compare
actual minimum box-corner height to a 0→12 cm reference over the original 2 s
lift ramp, and integrate their difference into a shared wrist-Z reference bias.
Gain is 1/s, bias rate ≤6 cm/s, bias range [0,8] cm. HOLD retains feedback;
LOWER smoothly removes the bias over 2 s; RELEASE has no added bias.

This is compensation for loaded tracking error, not a change to the required
real 8 cm / 2 s / stable replace-and-release sequence. Policy remains the same
official checkpoint; the planner still commands only sparse references, never
joint states, body forces, or object motion. If tilt, slip, contact, force or
time guards fail, the run fails and is retained. No success is presumed.

The option defaults to zero gain so the previous v4 targets can still be
reproduced. All runner/planner/config sources are snapshotted per run. The first
test is finite and unrecorded for diagnosis; useful behaviour is subsequently
recorded uncut and re-evaluated, never presented as the formal 60-case result.
