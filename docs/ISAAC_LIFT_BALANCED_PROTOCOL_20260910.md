# Native contact-only lift: centred sparse targets, v4

Fixed before execution on 2026-09-10. Development, single initial state/seed 42,
not formal D1/D2 acceptance. Original v1/v2/v3 inputs and failed runs are retained.

The native official model_22200.pt, robot-only FK seed, VR-3 mask, PhysX backend,
1 kg box, geometry, material, PD and torque limits remain unchanged. No training,
joint attachment, external force or post-initialization state write is permitted.

## Controller hypothesis

V3's wrist origins lagged issued X targets by about 6–7 cm and the box tipped
primarily in pitch. Contacts were not localized in that run; a rear-biased grasp
is a hypothesis, not yet a proven cause. V4 adds contact-point and friction-force
telemetry, separately labelled as the last physical substep, not four-step history.

`BalanceConfig` is snapshotted in each run and hashed with the planner. Nominal
wrist X moves from box X − 9 cm to X − 3 cm. Only during the last 2 s of the
3 s approach, a per-wrist X/Z tracking-bias integrator runs: gain 0.6/s, each
component rate ≤0.035 m/s, X bias ≤8 cm and Z bias ≤4 cm. Bias freezes before
contact closing. Actual wrist error is not a proxy for box task success.

At preload entry the measured box XY becomes a fixed tangential/side anchor;
targets no longer follow box drift during loaded phases. Existing independent
24 N side-force feedback and gap bounds are retained. Box pitch causes opposite
wrist pitch (gain 1.5, correction ≤0.25 rad); roll causes opposing wrist-height
correction (0.08 m/rad, ≤2.5 cm). Only sparse targets are changed; the existing
live provider's translational and angular speed/workspace limits still apply.

## Unchanged acceptance, stricter diagnostics

Both sides must still measure ≥18 N for 0.20 s before lift. Real lowest box
corner must clear the table by ≥8 cm, maintain that for ≥2 s, then return to
support and release stably for 1 s. Tilt ≤15°, relative slip ≤5 cm, allowed
contact bodies, 20 s total/phase deadlines and pose/velocity gates stay fixed.

In addition to the previous minimum-substep 40 N side-force protection, v4
rejects any of the four sampled side-force totals >40 N. Airborne clearance is
guarded by maximum support force across all four physics substeps (not minimum);
lower/release still require the conservative minimum support force. These are
measurement tightenings, not easier gates. The runner records both min and max.

`execution_complete` and `task_success` remain distinct. Full sequence success
in one development initial state is not the formal 60-episode criterion.

## Recording and lifecycle

Record every physical control step from frame 1 through terminal success/failure,
50 fps, without speed changes or removed failure segments. Initial reset-only
state is saved as JSON but not encoded because this installed renderer can show
stale reset visuals. Use the already tested native render-only warmup after
physical frame 1; verify robot/object/support state and physics counter unchanged
by rendering. Failure videos are explicitly labelled failed development trials.

Run only in finite `bfm-isaac-lift-*.service` user units, KillMode=control-group,
Restart=no, memory/tasks/time caps, one GPU task at a time, never kill other
projects. Preserve full trajectories, contact samples, input/policy SHA and final
result. Verify MainPID=0 and empty ControlGroup after every run. No GitHub push.

## Revision after trial d, before forward trial

Source revisions are identified by hashes and per-run `source_at_launch` copies;
prior runs are never overwritten. Strengthen release measurement to require each
side's **maximum** force over all four substeps below 0.2 N, rather than allowing
an intermittent contact to disappear in the minimum. This only affects RELEASE,
which trial d never reached. It cannot turn d's tilt failure into success.
