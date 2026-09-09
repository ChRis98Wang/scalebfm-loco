# Online VR-3 Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver opt-in continuous pelvis/wrist GUI reference control with no online resets and bounded real IsaacLab validation.

**Architecture:** A pure NumPy provider owns atomic target packets and continuous limited-rate future references. An explicit online-only MotionCommand subclass adapts it without changing the offline command, and the player owns supervision, GUI and cleanup. Existing policies remain unchanged.

**Tech Stack:** Existing Python 3.12 IsaacLab environment, NumPy, PyTorch, pytest and native Kit UI; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-07-online-vr3-reference-design.md`

## Global Constraints

- 首版限定单个 G1 29 DoF、平地、VR-3 掩码、固定世界轴方向的环境局部坐标。
- 沿用现有 IsaacLab 和权重格式，不重装依赖，不改变 PPO、网络结构或训练数据。
- GUI 默认 checkpoint 不因本功能而更换；新增 KIT 训练副本不自动升级为默认策略。
- 在线开始后不调用 `env.reset`、`switch_motion`、`switch_mode` 或机器人状态写接口。
- 首版不创建 socket、后台线程、常驻服务或另一个 GUI 进程。
- Tests use `/home/sw/isaaclab_ws/env_isaaclab_sim6_newton/bin/python`; numeric library threads are limited to 1.
- No push, uploads, raw-data edits, checkpoint promotion, physical object attachment or true-robot operation.
- Existing uncommitted files belong to the user. Commit new task files only; review integration edits against their pre-task contents, not as newly authored historical work.

## File boundaries and execution

1. `scaletrack/utils/live_reference.py`: immutable input, bounded pose trajectory and CPU sampling; no IsaacLab imports.
2. `scaletrack/utils/online_session.py`: state machine, heartbeat and actual-state health; no Kit imports.
3. `tracking/mdp/live_commands.py`: explicit `LiveMotionCommand(MotionCommand)`, installed only in online playback config; no changes to default command class.
4. `scripts/pretrain/rsl_rl/online_control.py`: CLI validation, config, runtime binding, per-frame supervision and telemetry.
5. `scripts/pretrain/rsl_rl/online_ui.py`: native Kit editing and deferred user intents, no physics writes.
6. Existing `play.py` and `scripts/open_motion_browser.sh`: opt-in wiring only; defaults unchanged.
7. Focused tests for provider/session/command/runtime/UI; a separate bounded real-GUI smoke entry point.

Execution is subagent-driven in the current session; controller owns review, integration checks and actual simulator invocations. The user asked to start development, so no additional execution-method approval is needed. Source edits run on a feature branch, never directly on main. Worktree preference is asked non-blockingly; without a request to create a worktree, retain the existing directory and user changes.

## Task 1: Atomic live reference provider

**Files:** Create `ScaleTrack/source/scaletrack/scaletrack/utils/live_reference.py` and `ScaleTrack/tests/test_live_reference.py`.

**Interfaces (produces):**

```python
VR3_BODY_NAMES = ("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link")
# LiveReferenceLimits: frozen dataclass, all finite/positive.
# pelvis_speed=.2, wrist_speed=.35, pelvis_angular_speed=.5,
# wrist_angular_speed=1., pelvis_radius=1., pelvis_height_delta=.15,
# pelvis_height_min=.35, pelvis_height_max=1.25, wrist_radius=1.

class LiveReferenceProvider:
    def __init__(self, body_names, body_pos, body_quat, joint_pos, *, step_dt=.02, limits=None): ...
    def submit(self, positions, quaternions, *, sequence, stamp, now,
               body_names=VR3_BODY_NAMES, frame="env_local", quaternion_order="wxyz"): ...
    def commit_pending(self, now): ...  # bool: accepted a pending complete packet
    def advance(self): ...            # exactly one step_dt; getters do not advance
    def sample(self, offsets): ...    # dict of independent NumPy arrays
    def reset_reference(self, body_pos, body_quat, joint_pos): ...
    def close(self): ...
```

Seed shapes: `(14,3)`, `(14,4)` canonical wxyz, `(29,)`; body names unique, VR3 resolves exactly to 0/10/13. Inputs and output buffers must not alias caller-owned arrays. `sample` accepts 1-D integer offsets in `[0,32]` and returns `body_pos`, `body_quat`, `body_lin_vel`, `body_ang_vel` with leading frame dimension plus `joint_pos/joint_vel` `(F,29)`. Public read-only metadata: `body_names`, `step_dt`, `limits`, `version`, `frame_index`, `last_sequence`, `rejected_count`, `last_error`; `goal_positions` and `goal_quaternions` return copies of the three current goal poses.

Sequences are integer, non-bool, strictly increasing across queued and accepted packets; timestamps finite, nonnegative, no future timestamp and no reversal relative to an earlier accepted/queued packet. Use `ReferenceInputError(ValueError)` with a `.clock_fault` flag for malformed time. Ordinary validation rejects the entire packet, retains previous pending/accepted packet, and counts rejection. Commit rechecks time without trusting mutable caller state. GUI clock and consumer use the same monotonic clock epoch.

Non-active body/joint references remain at seed; only three target bodies change. Current reference never jumps at commit. Future positions follow straight-line travel capped by distance; quaternions use shortest-arc interpolation capped by rotation angle. Predict frame 33 internally to calculate forward-difference velocities for public frame 32. Angular velocities are world-frame rotation-log differences, not Euler differences. Runtime quaternion conversion is not this module's job. Reset on resume reseeds reference, clears pending/old goal but retains original workspace bounds and sequence watermark. Close rejects further input.

- [x] Step 1: Add a failing behavioral test using `getattr(module, 'LiveReferenceProvider', None)` after checking file presence, so absence produces an intentional assertion instead of collection failure. A fixture uses pelvis `[0,0,.8]`, left/right wrists `[.2,.2,1.]` / `[.2,-.2,1.]`, identity wxyz quaternions and zero joints.

```python
provider.submit(np.array([[.2,0,.8],[.2,.2,1.],[.2,-.2,1.]]),
                np.tile([1.,0.,0.,0.], (3,1)), sequence=1, stamp=1., now=1.)
assert provider.commit_pending(1.)
np.testing.assert_allclose(provider.sample([0])["body_pos"][0,0], [0,0,.8])
np.testing.assert_allclose(provider.sample([32])["body_pos"][0,0], [.128,0,.8])
provider.advance()
np.testing.assert_allclose(provider.sample([0])["body_pos"][0,0], [.004,0,.8])
```

- [x] Step 2: Run `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/sw/isaaclab_ws/env_isaaclab_sim6_newton/bin/python -m pytest ScaleTrack/tests/test_live_reference.py -q`; record expected RED.
- [x] Step 3: Implement the contract with array validation, immutable copies, capped linear interpolation and quaternion interpolation; add test-first cases for 100 updates, velocity bounds/terminal frame, invalid inputs retaining last good packet, zero/nonunit/quaternion sign equivalence, independent buffers, seed/workspace validation, sequence/time ordering and resume/close.

```python
distance = np.linalg.norm(goal - current, axis=-1, keepdims=True)
fraction = np.minimum(1., speed * elapsed / np.maximum(distance, 1e-12))
position = current + fraction * (goal - current)
linear_velocity = (positions[1:] - positions[:-1]) / step_dt
```

- [x] Step 4: Run focused tests and the existing full relevant regression command from the spec; record GREEN and known dependency warnings separately.
- [x] Step 5: Commit only the two new task files and report exact test commands/results. Do not edit player/config/data in this task.

## Task 2: Online state machine and command adapter

**Files:** Create `ScaleTrack/source/scaletrack/scaletrack/utils/online_session.py`, `ScaleTrack/source/scaletrack/scaletrack/tasks/tracking/mdp/live_commands.py`, `ScaleTrack/tests/test_online_session.py`, `ScaleTrack/tests/test_live_commands.py`.

**Consumes:** Task 1 provider contract.

**Produces:**

```python
class OnlineSession:
    def __init__(self, provider, *, timeout=.5): ...
    def heartbeat(self, stamp, now): ...
    def activate(self, now, body_pos, body_quat, joint_pos): ...
    def pause(self, reason="user"): ...
    def fault(self, reason): ...
    def check(self, now, body_pos, body_quat, joint_pos, joint_vel): ...
    def resume(self, now, body_pos, body_quat, joint_pos): ...
    def close(self): ...
    # state: READY/ACTIVE/PAUSED_USER/PAUSED_FAULT/CLOSED; reason: str

class LiveMotionCommand(MotionCommand):
    def attach_live_reference(self, provider): ...
    # live_reference is None through initialization, then a provider.
```

State methods act only on references, never env/sim. `check` returns bool permitting a step only when ACTIVE, fresh heartbeat (age <=.5), finite real telemetry, pelvis z>=.35 and local-up dot world-up>=.5; otherwise latches fault. `activate` accepts READY only with valid heartbeat/telemetry. `resume` accepts paused states with fresh heartbeat/healthy telemetry, reseeds reference, preserves physical/history state and fixed workspace. Recoverable input faults require explicit resume; fall/nonfinite real-state faults cannot be silently resumed with fabricated pose. Invalid calls cannot implicitly enter ACTIVE. UI no-edit does not count as stale. Time corruption latches fault and cannot be cleared by ordinary heartbeat.

Live command inherits offline behavior while unattached. Once attached: override all current/manual/normal future body, anchor and joint reference getters; do not override `robot_*` actual getters. Convert provider wxyz once to runtime order, add origins once to positions, preserve dtype/device/shapes and 34-entry generic future including existing `_rand_timestep`. Cache by provider version/frame; negative manual `-1` resolves existing random offset, not a new RNG draw. `_update_command` advances provider once, not offline time. Online `compute(dt)` does metrics plus reference advancement with no timer resampling; `_resample_command` rejects online calls before any write. Attach validates one env, timestep/horizon/body order and installs VR-3 mask through existing configuration, never reset.

- [x] Step 1: Add failing tests for latched pause, stale heartbeat while no-edit remains valid, invalid resume and true-state fall; exercise real pure provider/session.

```python
session.heartbeat(1., 1.)
session.activate(1., body_pos, body_quat, joints)
assert session.check(1.4, body_pos, body_quat, joints, np.zeros(29))
assert not session.check(1.6, body_pos, body_quat, joints, np.zeros(29))
assert session.state == "PAUSED_FAULT"
session.heartbeat(1.7, 1.7)
assert session.state == "PAUSED_FAULT"
```

- [x] Step 2: Record RED using focused pytest command with both new test paths. Adapter tests may instantiate actual class via `__new__` with minimal articulation data, but must exercise actual property methods; no replacing the adapter itself with a mock.
- [x] Step 3: Implement session and explicit subclass, consolidating field sampling in one helper. Add tests for shifted origins, both quaternion runtime orders, frame32/manual/normal/anchor consistency, zero physics writes and no offline timeout after more than 500 advances.

```python
def _update_command(self):
    if self.live_reference is None:
        return super()._update_command()
    self.live_reference.advance()
    self._future_manual_cache.clear()
```

- [x] Step 4: Focused tests then full relevant regression, record GREEN.
- [x] Step 5: Commit only these new task files; no changes to offline command or existing physics configuration.

## Task 3: Player, GUI and launcher integration

**Files:** Create `ScaleTrack/scripts/pretrain/rsl_rl/online_control.py`, `ScaleTrack/scripts/pretrain/rsl_rl/online_ui.py`, `ScaleTrack/tests/test_online_control.py`, `ScaleTrack/tests/test_online_ui.py`. Modify `ScaleTrack/scripts/pretrain/rsl_rl/play.py`, `scripts/open_motion_browser.sh`, `tests/test_gui_launcher.py`.

**Consumes:** Provider, session and LiveMotionCommand interfaces above.

**Produces:** `add_online_args(parser)`, `validate_online_args(args)` (pure/pre-AppLauncher); `configure_online_env(env_cfg)` (sets subclass, removes three auto-termination terms and state-writing interval events only on this config); `OnlinePlaybackController(env, command, panel)` exposing `before_step(obs)`, `after_step(obs)`, `validate_actions(actions)` (bool), `close_requested` (bool), `close()`; `OnlineTargetPanel` with atomic complete pose Apply, `request_enable()`, `request_pause()`, `request_resume()`, `request_close()`, `heartbeat_stamp`, `consume_request()`, `update_status(status)` and `close()`.

`before_step` pumps/consumes intents and returns refreshed obs or None when physics must not run. `after_step` reads actual state after buffers update and checks health, updates telemetry, and returns no fake success flag. Panel production heartbeat comes from Kit update subscription, not from consumer code; test can stop subscription while consumer keeps running. The runtime catches clock/provider/policy faults into latched pause. Programmer/setup failures propagate after cleanup. GUI displays complete accepted goal, limited reference and actual articulation separately; pose editors use XYZ degrees to stable wxyz. On resume reset editors to actual pose and require new goal submission; do not replay stale pending intent.

Only `--online_targets --motion_menu --num_envs 1 --task G1-BFM-Transformer-Tracking` in Kit GUI is accepted. Default mode becomes 2 only online; conflicting explicit mode is rejected prelaunch. Reject local_tracking/video/headless and explicit non-Kit visualizer. Launcher `--online-targets` adds opt-in flag/mode2, keeps existing offline defaults, prevents implicit box resets. Online runtime disables legacy clip/mode/reset controls; offline restart requires closing/relaunching.

- [ ] Step 1: Add RED tests for argument rejection before app start and launcher dry-run output; tests execute parser/helper and actual shell dry-run, not source-text matches. Add controller fake external-boundary tests with real session/provider: 100 commits no reset/write/history append, fault freezes policy/physics, actual telemetry differs from goal, pending close is honored.

```python
args = argparse.Namespace(online_targets=True, motion_menu=True, num_envs=2,
                          task="G1-BFM-Transformer-Tracking", mode_index=None,
                          local_tracking=False, video=False, headless=False)
with pytest.raises(ValueError):
    validate_online_args(args)
```

- [ ] Step 2: Record RED focused tests, then implement early CLI validation, online-only config and a delegated online branch in the existing player loop.

```python
if online_controller is not None:
    candidate_obs = online_controller.before_step(obs)
    if online_controller.close_requested:
        break
    if candidate_obs is None:
        continue
    obs = candidate_obs
actions = policy(obs)
if online_controller is not None and not online_controller.validate_actions(actions):
    continue
obs, _, _, _ = env.step(actions)
if online_controller is not None:
    online_controller.after_step(obs)
```

Refresh only task groups via `compute_group(..., update_history=False)` before inference after commit; do not duplicate proprio/action history. Pause actual simulation and pump GUI without calling env.step. Native Kit Play cannot bypass fault latch. Register cleanup as soon as each object exists; no socket/thread created. Controller calls no physical reset and reads actual articulation for telemetry, never local_tracking reference alias.

- [ ] Step 3: Add GUI model tests using a lightweight UI boundary or actual Kit where needed. Test 3-target snapshot, degree conversion, pending coalescence and intent/close cleanup. Use existing marker APIs for three goal frames; renderer and physics stay owned by current process.
- [ ] Step 4: Run focused tests plus full regression. Preserve all earlier modifications to the three existing files. Keep their integration diff independently reviewable against pre-task snapshots; do not accidentally stage historical user changes.
- [ ] Step 5: Commit new task files only; controller arranges a scoped review package including the integration delta for already-dirty files. Leave unrelated files unstaged.

## Task 4: Bounded real-policy GUI acceptance and user documentation

**Files:** Create `ScaleTrack/tests/online_robot_gui_smoke.py` and `docs/ONLINE_VR3_CONTROL.md`; modify plan progress and relevant capability row after evidence is collected.

**Consumes:** Task 3 complete online path. Existing checkpoint is loaded strictly; no training.

**Produces:** GUI callback-driven acceptance entry point and report under a new `logs/online_control/` run directory, containing checkpoint/source/config hashes, input sequence, actual/reference trajectories, active position/orientation metrics, fault/pause state, reset/write/physics-step counters and cleanup exit status. Entry point hooks only observers and user intents, not fake policy/physics; no test-only methods in production.

- [ ] Step 1: Implement a test-first subprocess entry point that fails clearly if online classes are missing or interface assertions fail. Follow existing `playback_robot_gui_smoke.py` runpy player pattern. Initialize a small trusted train-only YAML to avoid loading entire corpus; do not modify old indices. Exact reference key/hash is recorded.

```python
before = torch.cat((env.scene["robot"].data.joint_pos,
                    env.scene["robot"].data.joint_vel), dim=-1).clone()
# Accumulate dt delivered by native PhysX post-step callbacks; verify ACTIVE progression first.
# This SDK's generic manager clock stays zero, and raw get_simulation_time is absent.
simulation_time_before = observed_physics_seconds
panel.request_pause()
# Drive normal Kit updates for >=2 wall-clock seconds, no env.step.
after = torch.cat((env.scene["robot"].data.joint_pos,
                   env.scene["robot"].data.joint_vel), dim=-1)
torch.testing.assert_close(after, before, rtol=0, atol=0)
assert observed_physics_seconds == simulation_time_before
panel.request_close()
```

- [ ] Step 2: Run each real GUI case in an owned transient systemd cgroup with RuntimeMaxSec=180, TimeoutStopSec=15, KillMode=control-group, no restart, existing Python and DISPLAY. Do not run parallel simulators. Record exit and ensure MainPID=0/inactive and no descendant GPU process.
- [ ] Step 3: Normal case attempts 100 small valid updates; separate bad-input, stale-heartbeat, pause/resume, close-during-fault and setup-failure cases validate shutdown. If policy falls before completing targets, record that limitation; do not disable fault checks or teleport to claim a pass.
- [ ] Step 4: Re-run full tests after any fixes, request full implementation review, then document actual GUI command, mode limits, tracking metrics and unmet physical-task gates. Existing offline checkpoint selection remains unchanged; measured tracking quality is not inferred from unit-test count.
- [ ] Step 5: Commit only reviewed new files/owned doc changes, no push; update completion boxes from evidence, not intent.
